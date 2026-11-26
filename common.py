"""
common.py - shared code for the Network Intrusion Detection System (NIDS) project.

Used by train.py (training), predict.py (command line) and app.py (web form).
Keeping the feature list, preprocessing and prediction helpers in ONE place
guarantees that new, unseen records go through exactly the same steps as the
training data.
"""
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TARGET = "label"            # 0 = normal, 1 = attack  (binary target)
ATTACK_CAT = "attack_cat"   # attack family  (used ONLY for analysis, never as a feature)
CLASS_NAMES = {0: "Normal Traffic", 1: "Malicious Traffic"}

# ----------------------------------------------------------------------------
# Feature definition
# ----------------------------------------------------------------------------
# Dropped on purpose (leakage / identifiers):
#   id                      -> row identifier
#   attack_cat, label       -> the target itself (attack_cat is target-derived)
#   srcip, dstip, sport,    -> identifiers of one specific host/connection; a model that
#   dsport, stime, ltime       memorises them would not generalise (only in the raw 2.5M files)
#   stcpb, dtcpb            -> random TCP initial sequence numbers (pure identifiers/noise)
#   rate                    -> only exists in the partition files, not in the 4 raw files,
#                              and is derived from dur/spkts/dpkts
NUMERIC = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload",
    "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth",
    "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]
CATEGORICAL = ["proto", "service", "state"]
FEATURES = NUMERIC + CATEGORICAL

# Fields shown in the command-line / web interface (all other features are
# filled with the training-set median / most frequent value).
FEATURE_INFO = {
    "proto":        ("Protocol", "tcp, udp, icmp, ..."),
    "service":      ("Service", "http, dns, ftp, ssh, '-' (none) ..."),
    "state":        ("Connection state", "FIN, CON, INT, REQ, RST ..."),
    "dur":          ("Duration (seconds)", "record total duration"),
    "spkts":        ("Source -> destination packets", "packet count"),
    "dpkts":        ("Destination -> source packets", "packet count"),
    "sbytes":       ("Source -> destination bytes", "bytes transferred"),
    "dbytes":       ("Destination -> source bytes", "bytes transferred"),
    "sttl":         ("Source TTL", "time-to-live, e.g. 31, 62, 254"),
    "dttl":         ("Destination TTL", "time-to-live, e.g. 29, 252, 0"),
    "sload":        ("Source bits per second", "source load"),
    "dload":        ("Destination bits per second", "destination load"),
    "smean":        ("Mean source packet size", "bytes"),
    "dmean":        ("Mean destination packet size", "bytes"),
    "ct_srv_src":   ("Connections, same service+source (last 100)", "count"),
    "ct_state_ttl": ("Connections per state/TTL range", "count 0-6"),
    "ct_dst_ltm":   ("Connections, same destination (last 100)", "count"),
}
KEY_FEATURES = list(FEATURE_INFO)

# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
# Column names of the 4 raw files UNSW-NB15_1..4.csv (these files have NO header row).
RAW_COLS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts",
    "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth",
    "res_bdy_len", "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt", "tcprtt",
    "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
    "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
    "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label",
]
RENAME = {"smeansz": "smean", "dmeansz": "dmean", "res_bdy_len": "response_body_len",
          "sintpkt": "sinpkt", "dintpkt": "dinpkt"}
KEEP = set(FEATURES + [TARGET, ATTACK_CAT])


def _canon(name):
    name = name.strip().lower()
    return RENAME.get(name, name)


def _clean_cat(s):
    s = s.astype("string").str.strip().str.lower()
    s = s.mask(s == "")
    return s.astype(object).where(s.notna(), np.nan)


def load_dataset(data_dir):
    """Read every UNSW-NB15 CSV in `data_dir` and return ONE clean DataFrame.

    Works with both versions of the dataset:
      * the 'partition' files  UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv (with header)
      * the 4 raw files        UNSW-NB15_1.csv ... UNSW-NB15_4.csv (no header, 49 columns)
    Only the columns we need are read, which saves a lot of memory.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. Download UNSW-NB15 and put the CSV "
            f"files there (see README.md).")
    frames = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            first = fh.readline()
        low = first.lower()
        if "label" in low and "proto" in low:                      # file with header
            df = pd.read_csv(f, usecols=lambda c: _canon(c) in KEEP, low_memory=False)
        elif first.count(",") + 1 == len(RAW_COLS):                # raw file, no header
            use = [c for c in RAW_COLS if _canon(c) in KEEP]
            df = pd.read_csv(f, header=None, names=RAW_COLS, usecols=use,
                             low_memory=False, encoding_errors="ignore")
        else:
            print(f"  [skip] {os.path.basename(f)} (not a UNSW-NB15 data file)")
            continue
        df.columns = [_canon(c) for c in df.columns]
        print(f"  loaded {os.path.basename(f):45s} {len(df):>9,} rows")
        frames.append(df)
    if not frames:
        raise ValueError("No usable UNSW-NB15 CSV files were found.")
    df = pd.concat(frames, ignore_index=True)
    return tidy(df)


def tidy(df):
    """Fix data types / text so every downstream step sees consistent values."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL:
        df[c] = _clean_cat(df[c])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype(int)
    if ATTACK_CAT in df.columns:
        ac = df[ATTACK_CAT].astype("string").str.strip()
        ac = ac.mask(ac.isna() | (ac == ""), "Normal").replace({"Backdoors": "Backdoor"})
        df[ATTACK_CAT] = ac.astype(object)
    else:
        df[ATTACK_CAT] = np.where(df[TARGET] == 1, "Attack", "Normal")
    return df[FEATURES + [ATTACK_CAT, TARGET]]


# ----------------------------------------------------------------------------
# Preprocessing + models
# ----------------------------------------------------------------------------
def signed_log1p(X):
    """Network counters (bytes, packets, load) are extremely right-skewed;
    log1p squeezes them so Logistic Regression / scaling behave well."""
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                 sparse_output=False, dtype=np.float32)),
    ])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def make_models():
    """The three required classifiers, each wrapped with the preprocessing step."""
    return {
        "Logistic Regression": Pipeline([
            ("pre", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
        "Decision Tree": Pipeline([
            ("pre", build_preprocessor()),
            ("clf", DecisionTreeClassifier(random_state=SEED))]),
        "Random Forest": Pipeline([
            ("pre", build_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1))]),
    }


def param_grids(fast=False):
    """Hyper-parameter grids for GridSearchCV ('clf__' = the classifier step)."""
    if fast:
        return {
            "Logistic Regression": {"clf__C": [0.1, 1, 10]},
            "Decision Tree": {"clf__max_depth": [8, 16, None], "clf__min_samples_leaf": [1, 5]},
            "Random Forest": {"clf__n_estimators": [50, 100], "clf__max_depth": [12, None]},
        }
    return {
        "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100],
                                "clf__class_weight": [None, "balanced"]},
        "Decision Tree": {"clf__max_depth": [8, 12, 16, None],
                          "clf__min_samples_leaf": [1, 5],
                          "clf__criterion": ["gini", "entropy"]},
        "Random Forest": {"clf__n_estimators": [100, 200],
                          "clf__max_depth": [12, 20, None],
                          "clf__min_samples_leaf": [1, 2]},
    }


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(model, X, y, threshold=0.5):
    """Return metrics (attack = positive class), predictions and attack probabilities."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "false_negative_rate": fn / max(fn + tp, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
    }
    return m, pred, proba


def text_report(y, pred):
    return classification_report(y, pred, target_names=["Normal", "Attack"], digits=4)


# ----------------------------------------------------------------------------
# Saving / loading / predicting on new records
# ----------------------------------------------------------------------------
def build_metadata(X_train):
    """Information the interface needs: defaults for missing fields + allowed categories."""
    defaults = {c: float(X_train[c].median()) for c in NUMERIC}
    choices = {}
    for c in CATEGORICAL:
        vc = X_train[c].value_counts()
        defaults[c] = str(vc.index[0])
        choices[c] = [str(v) for v in vc.index[:60]]
    return {"features": FEATURES, "numeric": NUMERIC, "categorical": CATEGORICAL,
            "key_features": KEY_FEATURES, "feature_info": FEATURE_INFO,
            "defaults": defaults, "choices": choices, "class_names": CLASS_NAMES}


def load_artifacts(model_dir):
    path = os.path.join(model_dir, "best_model.joblib")
    meta_path = os.path.join(model_dir, "metadata.json")
    if not (os.path.exists(path) and os.path.exists(meta_path)):
        raise FileNotFoundError(
            f"No trained model found in '{model_dir}'. Run  python train.py  first.")
    with open(meta_path) as fh:
        meta = json.load(fh)
    return joblib.load(path), meta


def records_to_frame(records, meta):
    """Turn user input (dicts) into a DataFrame with ALL model features.
    Fields the user did not supply get the training-set default."""
    rows = []
    for rec in records:
        row = dict(meta["defaults"])
        for key, value in rec.items():
            k = str(key).strip().lower()
            if k not in FEATURES or value is None or str(value).strip() == "":
                continue
            if k in CATEGORICAL:
                row[k] = str(value).strip().lower()
            else:
                try:
                    num = float(value)
                except ValueError:
                    raise ValueError(f"'{k}' must be a number, but got {value!r}")
                if not np.isfinite(num):
                    raise ValueError(f"'{k}' must be a finite number")
                row[k] = num
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURES)


def predict_records(records, model, meta, threshold=0.5):
    """records: list of dicts -> list of result dicts."""
    X = records_to_frame(records, meta)
    proba = model.predict_proba(X)[:, 1]
    out = []
    for p in proba:
        label = int(p >= threshold)
        out.append({"label": label, "prediction": CLASS_NAMES[label],
                    "attack_probability": float(p)})
    return out
