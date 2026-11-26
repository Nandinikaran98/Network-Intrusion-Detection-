#!/usr/bin/env python3
"""
train.py - Network Intrusion Detection (UNSW-NB15) : full training pipeline.

Runs project steps 1-9 and 11 (saving):
  1 load  2 EDA  3 prepare  4 split  5 train 3 models  6 evaluate  7 5-fold CV
  8 GridSearchCV  9 select best + one final evaluation on the untouched test set
  11 save pipeline + metadata

Usage:
    python train.py --data-dir data                 # real dataset (a few minutes)
    python train.py --data-dir data_sample --fast   # quick demo on synthetic data
"""
import argparse
import json
import os
import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_validate,
                                     train_test_split)

import common as C

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"Normal": "#2a9d8f", "Attack": "#e63946"}


# ----------------------------------------------------------------------------- helpers
class Out:
    """Keeps output folders + collects the lines of the auto-generated report."""

    def __init__(self, root):
        self.root = root
        self.fig_dir = os.path.join(root, "figures")
        self.tab_dir = os.path.join(root, "tables")
        self.model_dir = os.path.join(root, "models")
        for d in (self.fig_dir, self.tab_dir, self.model_dir):
            os.makedirs(d, exist_ok=True)
        self.report = []

    def fig(self, name):
        path = os.path.join(self.fig_dir, name)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"   figure saved -> {path}")
        return f"figures/{name}"

    def table(self, df, name, index=True):
        df.to_csv(os.path.join(self.tab_dir, name), index=index)

    def add(self, text=""):
        self.report.append(text)


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def md_table(df, floatfmt="{:.4f}", index=True):
    d = df.reset_index() if index else df.copy()
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = []
    for _, r in d.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


def set_rf_jobs(pipe, n):
    if "clf__n_jobs" in pipe.get_params():
        pipe.set_params(clf__n_jobs=n)
    return pipe


def stratified_subset(X, y, n, seed):
    if n >= len(X):
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return Xs, ys


# ----------------------------------------------------------------------------- steps
def step1_load(args, out):
    banner("STEP 1 - Load and understand the dataset")
    df = C.load_dataset(args.data_dir)
    if args.max_rows and len(df) > args.max_rows:
        df, _ = train_test_split(df, train_size=args.max_rows, stratify=df[C.TARGET],
                                 random_state=args.seed)
        df = df.reset_index(drop=True)
        print(f"   --max-rows: using a stratified sample of {len(df):,} rows")
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Input features used: {len(C.FEATURES)} "
          f"({len(C.NUMERIC)} numeric + {len(C.CATEGORICAL)} categorical)")
    print(f"Target column: '{C.TARGET}' (0 = Normal, 1 = Attack); attack family column: '{C.ATTACK_CAT}'")

    info = pd.DataFrame({"dtype": df.dtypes.astype(str), "non_null": df.notna().sum(),
                         "missing": df.isna().sum(), "unique_values": df.nunique()})
    print("\nColumn summary:\n", info.to_string())
    out.table(info, "01_column_summary.csv")

    counts = df[C.TARGET].map(C.CLASS_NAMES).value_counts().rename("records").to_frame()
    counts["percent"] = (counts["records"] / len(df) * 100).round(2)
    cats = df[C.ATTACK_CAT].value_counts().rename("records").to_frame()
    cats["percent"] = (cats["records"] / len(df) * 100).round(2)
    print("\nTarget class counts:\n", counts.to_string())
    print("\nAttack category counts:\n", cats.to_string())
    out.table(counts, "01_target_class_counts.csv")
    out.table(cats, "01_attack_category_counts.csv")

    out.add("## 1. Dataset")
    out.add(f"- Records loaded: **{len(df):,}**; input features: **{len(C.FEATURES)}** "
            f"({len(C.NUMERIC)} numeric, {len(C.CATEGORICAL)} categorical)")
    out.add("- Target: `label` (0 = Normal, 1 = Attack); `attack_cat` is used only for analysis.\n")
    out.add(md_table(counts))
    out.add("")
    out.add(md_table(cats))
    return df


def step2_eda(df, out):
    banner("STEP 2 - Exploratory data analysis")
    out.add("\n## 2. Exploratory data analysis")

    # --- duplicates & missing values
    n_dup = int(df.duplicated().sum())
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    print(f"Duplicate records: {n_dup:,} ({n_dup / len(df):.2%})")
    print("Missing values per column:", "none" if miss.empty else "\n" + miss.to_string())
    out.add(f"- Duplicate records: **{n_dup:,}** ({n_dup / len(df):.2%}) - removed (see step 3).")
    out.add("- Missing values: " + ("none" if miss.empty else
            ", ".join(f"`{k}` ({v:,})" for k, v in miss.items())) +
            " - handled by imputers inside the pipeline.")

    # --- class imbalance
    vc = df[C.TARGET].value_counts()
    ratio = vc.max() / vc.min()
    cat_counts = df[C.ATTACK_CAT].value_counts()
    atk = cat_counts.drop("Normal", errors="ignore")          # attack families only
    fam_ratio = atk.max() / max(atk.min(), 1)
    print(f"\nNormal vs attack (majority/minority) ratio = {ratio:.2f} : 1")
    print(f"Largest attack family : {atk.idxmax()} ({atk.max():,}); "
          f"smallest: {atk.idxmin()} ({atk.min():,}); ratio {fam_ratio:.0f} : 1")
    out.add(f"- Normal vs attack (majority/minority) ratio: **{ratio:.2f} : 1**.")
    out.add(f"- Attack families: largest `{atk.idxmax()}` = {atk.max():,} records, smallest "
            f"`{atk.idxmin()}` = {atk.min():,} records (**{fam_ratio:.0f} : 1**) - "
            f"{'strongly imbalanced' if fam_ratio >= 10 else 'moderately imbalanced'}, so recall of "
            "the rare families is checked separately in the final evaluation.\n")

    # --- numeric & categorical summary tables
    desc = df[C.NUMERIC].describe().T
    out.table(desc, "02_numeric_summary.csv")
    cat_summary = pd.DataFrame({c: [df[c].nunique(), df[c].value_counts().index[0]] for c in C.CATEGORICAL},
                               index=["unique_values", "most_frequent"]).T
    out.table(cat_summary, "02_categorical_summary.csv")
    print("\nNumeric summary (first rows):\n", desc.head(8).round(3).to_string())
    print("\nCategorical summary:\n", cat_summary.to_string())

    # --- figures
    labels = df[C.TARGET].map({0: "Normal", 1: "Attack"})
    fig, ax = plt.subplots(figsize=(4.5, 4))
    lc = labels.value_counts()
    bars = ax.bar(lc.index, lc.values, color=[COLORS[k] for k in lc.index])
    ax.bar_label(bars, labels=[f"{v:,}\n({v / len(df):.1%})" for v in lc.values])
    ax.set_title("Normal vs attack traffic")
    ax.set_ylabel("records")
    ax.set_ylim(0, lc.max() * 1.18)
    f1 = out.fig("fig01_class_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    cc = cat_counts.sort_values()
    bars = ax.barh(cc.index, cc.values, color=["#2a9d8f" if k == "Normal" else "#e63946" for k in cc.index])
    ax.bar_label(bars, labels=[f"{v:,}" for v in cc.values], padding=3)
    ax.set_title("Traffic categories (Normal + attack families)")
    ax.set_xlabel("records")
    ax.set_xlim(0, cc.max() * 1.15)
    f2 = out.fig("fig02_attack_categories.png")

    show = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "sload"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, col in zip(axes.ravel(), show):
        for name, g in df.groupby(labels):
            ax.hist(np.log1p(g[col].clip(lower=0)), bins=40, alpha=0.55, label=name,
                    color=COLORS[name], density=True)
        ax.set_title(f"log(1 + {col})")
        ax.set_yticks([])
    axes[0, 0].legend()
    fig.suptitle("Distribution of key numeric features by class", y=1.01)
    f3 = out.fig("fig03_numeric_distributions.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, C.CATEGORICAL):
        top = df[col].value_counts().head(8).index
        tab = pd.crosstab(df[col].where(df[col].isin(top), "other"), labels)
        tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]
        tab.plot(kind="bar", stacked=True, ax=ax, color=[COLORS[c] for c in tab.columns], width=0.8)
        ax.set_title(f"'{col}' (top 8 values)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    f4 = out.fig("fig04_categorical_features.png")

    corr = df[C.NUMERIC].corr().fillna(0)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    ax.set_title("Correlation between numeric features")
    fig.colorbar(im, ax=ax, shrink=0.8)
    f5 = out.fig("fig05_correlation_heatmap.png")

    for f, cap in [(f1, "Class distribution"), (f2, "Traffic categories"),
                   (f3, "Numeric feature distributions"), (f4, "Categorical features"),
                   (f5, "Correlation heatmap")]:
        out.add(f"![{cap}]({f})")


def step3_prepare(df, out):
    banner("STEP 3 - Prepare the data")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df):,} duplicate records -> {len(df):,} left")
    X = df[C.FEATURES].copy()
    y = df[C.TARGET].copy()
    cats = df[C.ATTACK_CAT].copy()
    print(f"X: {X.shape}, y: {y.shape}")
    print("Leakage control: 'id', 'attack_cat', 'label', IPs/ports/timestamps and TCP sequence "
          "numbers are NOT in X.")
    print("Encoding (one-hot) and scaling (log + standardise) happen INSIDE the sklearn pipeline, "
          "so they are learned from training folds only.")
    out.add("\n## 3. Data preparation")
    out.add(f"- Removed {before - len(df):,} duplicates; final size **{len(df):,}** records.")
    out.add("- Features `X` exclude `id`, `attack_cat`, `label`, addresses/ports/timestamps and TCP "
            "sequence numbers (identifiers / target-derived columns -> leakage).")
    out.add("- Numeric: median imputation -> signed log1p -> StandardScaler. Categorical "
            "(`proto`, `service`, `state`): most-frequent imputation -> one-hot encoding.")
    return X, y, cats


def step4_split(X, y, cats, args, out):
    banner("STEP 4 - Split the dataset (stratified)")
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
        X, y, cats, test_size=args.test_size, stratify=y, random_state=args.seed)
    # an extra validation split INSIDE the training data: used to compare / select models,
    # so that the real test set is touched only once at the very end.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=args.seed)
    rows = pd.DataFrame({
        "records": [len(X_fit), len(X_val), len(X_test)],
        "attack_share": [y_fit.mean(), y_val.mean(), y_test.mean()]},
        index=["train-fit (60%)", "validation (20%)", "test (20%, untouched)"])
    print(rows.to_string())
    out.table(rows, "04_split_sizes.csv")
    out.add("\n## 4. Train / validation / test split")
    out.add(f"- {1 - args.test_size:.0%} train / {args.test_size:.0%} test (stratified on the label). "
            "The training part is divided again into fit (75%) and validation (25%) sets. "
            "Models are compared and tuned with train/validation data only; "
            "the test set is evaluated **once** at the end.\n")
    out.add(md_table(rows))
    return X_train, X_test, y_train, y_test, c_test, X_fit, X_val, y_fit, y_val


def step5_6_baseline(models, X_fit, y_fit, X_val, y_val, out):
    banner("STEP 5 - Train the three models (default settings)")
    trained, rows = {}, {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, pipe) in zip(axes, models.items()):
        t0 = time.time()
        pipe = clone(pipe).fit(X_fit, y_fit)
        trained[name] = pipe
        m, pred, _ = C.evaluate(pipe, X_val, y_val)
        m["train_seconds"] = time.time() - t0
        rows[name] = m
        print(f"   {name:20s} trained in {m['train_seconds']:.1f}s")
        ConfusionMatrixDisplay.from_predictions(
            y_val, pred, display_labels=["Normal", "Attack"], cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(name)
    fig.suptitle("Baseline models - confusion matrices (validation set)")
    f = out.fig("fig06_baseline_confusion_matrices.png")

    banner("STEP 6 - Evaluate and compare the models (validation set)")
    tbl = pd.DataFrame(rows).T
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "TN", "FP", "FN", "TP",
            "false_negative_rate"]
    tbl = tbl.astype({c: int for c in ("TN", "FP", "FN", "TP")} | {"train_seconds": float})
    print(tbl[cols].to_string(float_format=lambda v: f"{v:.4f}"))
    out.table(tbl[cols], "06_baseline_comparison.csv")
    for name, pipe in trained.items():
        _, pred, _ = C.evaluate(pipe, X_val, y_val)
        print(f"\n--- {name} ---\n{C.text_report(y_val, pred)}")
    best = tbl["recall"].idxmax()
    print(f"Highest attack recall (fewest missed attacks): {best}")
    out.add("\n## 5-6. Three models (default settings) - validation set")
    out.add("Attack = positive class. **Recall** and **FN** (attacks that slipped through) matter most.\n")
    out.add(md_table(tbl[cols]))
    out.add(f"\n![Baseline confusion matrices]({f})")
    return trained, tbl


def step7_cv(models, X_sub, y_sub, out, seed):
    banner(f"STEP 7 - 5-fold cross-validation (on {len(X_sub):,} training rows)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}
    fold_rows, summary = [], {}
    for name, pipe in models.items():
        pipe = set_rf_jobs(clone(pipe), 1)
        res = cross_validate(pipe, X_sub, y_sub, cv=skf, scoring=scoring, n_jobs=-1)
        for k in range(5):
            fold_rows.append({"model": name, "fold": k + 1,
                              **{m: res[f"test_{m}"][k] for m in scoring}})
        summary[name] = {f"{m}_mean": res[f"test_{m}"].mean() for m in scoring} | \
                        {f"{m}_std": res[f"test_{m}"].std() for m in ("f1",)}
        print(f"\n{name}\n   F1 per fold : {np.round(res['test_f1'], 4)}\n"
              f"   Recall/fold : {np.round(res['test_recall'], 4)}\n"
              f"   mean F1 = {res['test_f1'].mean():.4f} (+/- {res['test_f1'].std():.4f}), "
              f"mean accuracy = {res['test_accuracy'].mean():.4f}")
    folds = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary).T
    out.table(folds, "07_cv_fold_scores.csv", index=False)
    out.table(summary, "07_cv_summary.csv")
    out.add("\n## 7. 5-fold cross-validation (default models)")
    out.add(f"Run on a stratified training subset of {len(X_sub):,} rows.\n")
    piv = folds.pivot(index="fold", columns="model", values="f1")
    piv.loc["mean"] = piv.mean()
    out.add("F1-score of each fold:\n")
    out.add(md_table(piv))
    out.add("\nMean scores:\n")
    out.add(md_table(summary))
    return summary


def step8_tune(models, X_sub, y_sub, args, out):
    banner(f"STEP 8 - Hyper-parameter tuning with GridSearchCV (on {len(X_sub):,} training rows)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    grids = C.param_grids(args.fast)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}
    best_params, rows, all_results = {}, {}, []
    for name, pipe in models.items():
        t0 = time.time()
        gs = GridSearchCV(set_rf_jobs(clone(pipe), 1), grids[name], cv=skf, scoring=scoring,
                          refit="f1", n_jobs=-1)
        gs.fit(X_sub, y_sub)
        i = gs.best_index_
        best_params[name] = gs.best_params_
        rows[name] = {"best_params": json.dumps(gs.best_params_, default=str),
                      "cv_f1": gs.cv_results_["mean_test_f1"][i],
                      "cv_recall": gs.cv_results_["mean_test_recall"][i],
                      "cv_precision": gs.cv_results_["mean_test_precision"][i],
                      "cv_accuracy": gs.cv_results_["mean_test_accuracy"][i],
                      "combinations": len(gs.cv_results_["params"])}
        res = pd.DataFrame(gs.cv_results_)[["params", "mean_test_f1", "mean_test_recall",
                                            "mean_test_accuracy", "rank_test_f1"]]
        res.insert(0, "model", name)
        all_results.append(res)
        print(f"\n{name}  ({rows[name]['combinations']} combinations x 5 folds, {time.time() - t0:.0f}s)"
              f"\n   best params : {gs.best_params_}\n   best CV F1  : {gs.best_score_:.4f}"
              f"   (CV recall {rows[name]['cv_recall']:.4f})")
    tbl = pd.DataFrame(rows).T
    out.table(tbl, "08_best_hyperparameters.csv")
    out.table(pd.concat(all_results), "08_grid_search_all_results.csv", index=False)
    with open(os.path.join(out.tab_dir, "08_best_params.json"), "w") as fh:
        json.dump(best_params, fh, indent=2, default=str)
    out.add("\n## 8. Hyper-parameter tuning (GridSearchCV, 5-fold, scoring = F1 of the attack class)")
    out.add(md_table(tbl, floatfmt="{:.4f}"))
    return best_params, tbl


def step9_select_and_test(models, best_params, tuned_cv, X_fit, y_fit, X_val, y_val,
                          X_train, y_train, X_test, y_test, c_test, args, out):
    banner("STEP 9 - Select the best model")
    tuned, rows = {}, {}
    for name, pipe in models.items():
        p = clone(pipe).set_params(**best_params[name])
        p = set_rf_jobs(p, -1).fit(X_fit, y_fit)
        tuned[name] = p
        m, _, _ = C.evaluate(p, X_val, y_val)
        rows[name] = m
        joblib.dump(p, os.path.join(out.model_dir, f"tuned_{name.lower().replace(' ', '_')}.joblib"),
                    compress=3)
    val = pd.DataFrame(rows).T.astype({"FN": int})
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "FN", "false_negative_rate"]
    val["cv_f1"] = tuned_cv["cv_f1"].astype(float)
    print("Tuned models on the VALIDATION set:\n",
          val[cols + ["cv_f1"]].to_string(float_format=lambda v: f"{v:.4f}"))
    out.table(val[cols + ["cv_f1"]], "09_tuned_comparison_validation.csv")

    # selection rule: highest validation F1; recall is the tie-breaker
    best_name = val.sort_values(["f1", "recall"], ascending=False).index[0]
    print(f"\n>>> Selected model: {best_name}  (highest validation F1 of the attack class)")
    out.add("\n## 9. Tuned models and model selection (validation set)")
    out.add(md_table(val[cols + ["cv_f1"]]))
    out.add(f"\n**Selected model: {best_name}** - rule: highest validation F1-score of the attack class "
            "(recall as tie-breaker).")

    # comparison chart: baseline vs tuned
    return val, best_name, tuned


def final_test(models, best_params, best_name, X_train, y_train, X_test, y_test, c_test, out):
    banner("FINAL - Re-train the selected model on all training data, test ONCE")
    best = clone(models[best_name]).set_params(**best_params[best_name])
    best = set_rf_jobs(best, -1).fit(X_train, y_train)
    m, pred, proba = C.evaluate(best, X_test, y_test)
    print(f"Test set size: {len(X_test):,}")
    print(C.text_report(y_test, pred))
    print(f"Confusion matrix: TN={m['TN']:,} FP={m['FP']:,} FN={m['FN']:,} TP={m['TP']:,}")
    print(f"Missed attacks (false negatives): {m['FN']:,} = {m['false_negative_rate']:.2%} of attacks")
    out.table(pd.DataFrame([m]).T.rename(columns={0: "value"}), "10_final_test_metrics.csv")
    with open(os.path.join(out.tab_dir, "10_final_classification_report.txt"), "w") as fh:
        fh.write(f"Model: {best_name}\nParams: {best_params[best_name]}\n\n{C.text_report(y_test, pred)}")

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Normal", "Attack"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{best_name} - test set")
    f_cm = out.fig("fig07_final_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(5, 4.4))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, plot_chance_level=True,
                                     name=f"{best_name} (AUC = {m['roc_auc']:.4f})")
    ax.set_title("ROC curve - test set")
    f_roc = out.fig("fig08_roc_curve.png")

    # detection rate per traffic category
    per = pd.DataFrame({"cat": c_test.values, "pred": pred}).groupby("cat")["pred"].agg(["mean", "size"])
    per.columns = ["flagged_as_attack", "records"]
    per.loc["Normal", "flagged_as_attack"] = 1 - per.loc["Normal", "flagged_as_attack"]
    per = per.rename(index={"Normal": "Normal (correctly passed)"}).sort_values("flagged_as_attack")
    out.table(per, "10_per_category_detection.csv")
    print("\nDetection rate per category:\n", per.to_string(float_format=lambda v: f"{v:.4f}"))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.barh(per.index, per["flagged_as_attack"], color="#457b9d")
    ax.bar_label(bars, labels=[f"{v:.1%} (n={n:,})" for v, n in zip(per["flagged_as_attack"], per["records"])],
                 padding=3, fontsize=8)
    ax.set_xlim(0, 1.25)
    ax.set_title("Share of each category classified correctly - test set")
    f_cat = out.fig("fig09_per_category_detection.png")

    f_imp = None
    try:
        names = best.named_steps["pre"].get_feature_names_out()
        clf = best.named_steps["clf"]
        imp = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_[0])
        s = pd.Series(imp, index=[n.split("__", 1)[-1] for n in names]).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.barh(s.index, s.values, color="#6a4c93")
        ax.set_title(f"Top 15 features - {best_name}")
        f_imp = out.fig("fig10_feature_importance.png")
        out.table(s.sort_values(ascending=False).to_frame("importance"), "10_top_features.csv")
    except Exception as exc:                                   # pragma: no cover
        print("   (feature importance skipped:", exc, ")")

    out.add("\n## 10. Final evaluation of the selected model on the untouched test set")
    out.add(f"**Model:** {best_name}  \n**Hyper-parameters:** `{best_params[best_name]}`\n")
    out.add("```\n" + C.text_report(y_test, pred) + "```")
    out.add(f"\nConfusion matrix: TN = {m['TN']:,}, FP = {m['FP']:,}, FN = {m['FN']:,}, TP = {m['TP']:,}. "
            f"Missed attacks (false negatives): **{m['FN']:,}** ({m['false_negative_rate']:.2%} of all attacks); "
            f"false alarms: {m['FP']:,} ({m['false_positive_rate']:.2%} of normal traffic).\n")
    out.add(f"![Confusion matrix]({f_cm}) ![ROC curve]({f_roc})")
    out.add(f"\n![Per-category detection]({f_cat})")
    if f_imp:
        out.add(f"\n![Feature importance]({f_imp})")
    return best, m, pred


def comparison_chart(base, tuned_val, best_name, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        x = np.arange(len(base))
        ax.bar(x - 0.2, base[metric].astype(float), 0.4, label="default", color="#a8dadc")
        ax.bar(x + 0.2, tuned_val[metric].astype(float), 0.4, label="tuned", color="#1d3557")
        ax.set_xticks(x, base.index, rotation=15)
        ax.set_title(f"Attack {metric} (validation)")
        lo = min(base[metric].min(), tuned_val[metric].min())
        ax.set_ylim(max(0, lo - 0.05), 1.0)
    axes[0].legend()
    return out.fig("fig11_model_comparison.png")


def save_final(best, best_name, best_params, X_train, X_test, y_test, c_test, metrics, args, out):
    banner("STEP 11 - Save the complete pipeline")
    joblib.dump(best, os.path.join(out.model_dir, "best_model.joblib"), compress=3)
    meta = C.build_metadata(X_train)
    # two real (correctly classified) test examples for the demo buttons of the web interface
    # Examples are checked exactly the way the interface uses them (only the key features are
    # sent, all other features get the training default) and the most confident correct ones are kept.
    rng = np.random.default_rng(args.seed)
    examples = {}
    for lab, key in ((0, "normal"), (1, "attack")):
        idx = np.where(y_test.values == lab)[0]
        idx = rng.choice(idx, size=min(300, len(idx)), replace=False)
        recs = [{k: (str(X_test.iloc[i][k]) if k in C.CATEGORICAL else float(X_test.iloc[i][k]))
                 for k in C.KEY_FEATURES} for i in idx]
        res = C.predict_records(recs, best, meta)
        ok = [(r["attack_probability"] if lab else 1 - r["attack_probability"], rec)
              for r, rec in zip(res, recs) if r["label"] == lab]
        if ok:
            examples[key] = max(ok, key=lambda t: t[0])[1]
    meta.update({"model_name": best_name, "best_params": json.loads(json.dumps(best_params[best_name], default=str)),
                 "test_metrics": {k: float(v) for k, v in metrics.items()},
                 "examples": examples, "trained_at": datetime.now().isoformat(timespec="seconds"),
                 "train_rows": int(len(X_train))})
    with open(os.path.join(out.model_dir, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved: {out.model_dir}/best_model.joblib  (+ metadata.json, tuned_*.joblib)")
    out.add("\n## 11. Saved artefacts")
    out.add("- `models/best_model.joblib` - complete pipeline (preprocessing + classifier)\n"
            "- `models/tuned_*.joblib` - the three tuned models\n"
            "- `models/metadata.json` - feature list, defaults, categories, test metrics\n"
            "- Use `python predict.py` (CLI) or `python app.py` (web form) for new records.")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="folder with the UNSW-NB15 CSV files")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--tune-sample", type=int, default=30000,
                    help="rows of the training data used for 5-fold CV and GridSearchCV")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="optional: stratified sample of the whole dataset (for the 2.5M-row version)")
    ap.add_argument("--fast", action="store_true", help="smaller grids (quick demo)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    t_start = time.time()

    out = Out(args.out_dir)
    out.add("# Network Intrusion Detection - Results Report")
    out.add(f"_Generated automatically by train.py on {datetime.now():%Y-%m-%d %H:%M}_\n")

    df = step1_load(args, out)
    step2_eda(df, out)
    X, y, cats = step3_prepare(df, out)
    del df
    X_train, X_test, y_train, y_test, c_test, X_fit, X_val, y_fit, y_val = step4_split(X, y, cats, args, out)

    models = C.make_models()
    _, base_tbl = step5_6_baseline(models, X_fit, y_fit, X_val, y_val, out)

    X_sub, y_sub = stratified_subset(X_fit, y_fit, args.tune_sample, args.seed)
    step7_cv(models, X_sub, y_sub, out, args.seed)
    best_params, tuned_cv = step8_tune(models, X_sub, y_sub, args, out)
    val, best_name, _ = step9_select_and_test(models, best_params, tuned_cv, X_fit, y_fit, X_val, y_val,
                                              X_train, y_train, X_test, y_test, c_test, args, out)
    f_cmp = comparison_chart(base_tbl, val, best_name, out)
    out.add(f"\n![Baseline vs tuned]({f_cmp})")
    best, m, _ = final_test(models, best_params, best_name, X_train, y_train, X_test, y_test, c_test, out)
    save_final(best, best_name, best_params, X_train, X_test, y_test, c_test, m, args, out)

    with open(os.path.join(args.out_dir, "report.md"), "w") as fh:
        fh.write("\n".join(out.report))
    banner(f"DONE in {(time.time() - t_start) / 60:.1f} min - results in '{args.out_dir}/' "
           f"(report.md, figures/, tables/, models/)")


if __name__ == "__main__":
    main()
