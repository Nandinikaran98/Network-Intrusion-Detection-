#!/usr/bin/env python3
"""
predict.py - command-line Network Intrusion Detection System.

Classifies new network records with the saved pipeline (no re-training needed).

    python predict.py                          # interactive: type feature values
    python predict.py --example attack         # run a real attack record from the test set
    python predict.py --example normal
    python predict.py --csv new_records.csv    # batch mode -> writes new_records_predictions.csv
    python predict.py --json '{"proto": "tcp", "sbytes": 5000, "dur": 0.2}'

Any feature you leave out is filled with the training-set median / most frequent value.
"""
import argparse
import json
import os
import sys

import pandas as pd

import common as C

LINE = "=" * 60


def show(result, record=None):
    print("\n" + LINE)
    print("  Predicted class :", result["prediction"].upper())
    print(f"  Attack probability: {result['attack_probability']:.2%}")
    print(LINE)


def interactive(model, meta):
    print(LINE + "\n  Network Intrusion Detection System\n" + LINE)
    print("Enter the traffic features. Press ENTER to use the shown default value.\n")
    record = {}
    for key in meta["key_features"]:
        label, hint = meta["feature_info"][key]
        default = meta["defaults"][key]
        if key in meta["categorical"]:
            options = ", ".join(meta["choices"][key][:12])
            prompt = f"{label} [{default}] (e.g. {options}): "
        else:
            prompt = f"{label} [{default:g}] ({hint}): "
        value = input(prompt).strip()
        if value:
            record[key] = value
    try:
        show(C.predict_records([record], model, meta)[0])
    except ValueError as exc:
        print("Input error:", exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.path.join("outputs", "models"))
    ap.add_argument("--csv", help="CSV file with new records (columns named like the dataset)")
    ap.add_argument("--json", help="one record as a JSON string")
    ap.add_argument("--example", choices=["normal", "attack"], help="use a stored real example")
    args = ap.parse_args()

    try:
        model, meta = C.load_artifacts(args.model_dir)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    try:
        if args.csv:
            df = pd.read_csv(args.csv)
            results = C.predict_records(df.to_dict("records"), model, meta)
            df["prediction"] = [r["prediction"] for r in results]
            df["attack_probability"] = [round(r["attack_probability"], 4) for r in results]
            out = os.path.splitext(args.csv)[0] + "_predictions.csv"
            df.to_csv(out, index=False)
            print(df["prediction"].value_counts().to_string())
            print(f"\nPredictions saved to {out}")
        elif args.json:
            show(C.predict_records([json.loads(args.json)], model, meta)[0])
        elif args.example:
            record = meta["examples"][args.example]
            print("Input record:", json.dumps(record, indent=1))
            show(C.predict_records([record], model, meta)[0])
        else:
            interactive(model, meta)
    except (ValueError, FileNotFoundError) as exc:      # bad number, bad JSON, missing file ...
        sys.exit(f"Input error: {exc}")


if __name__ == "__main__":
    main()
