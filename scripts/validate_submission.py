#!/usr/bin/env python3
"""Check that a predictions.csv is accepted by the scorer.

    python scripts/validate_submission.py --pred predictions.csv --test data/test.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]
REQUIRED = ["run_id", "label", "success", "fault_turn"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()

    pred = pd.read_csv(a.pred)
    test = pd.read_csv(a.test, usecols=["run_id"])
    errors = []

    missing = [c for c in REQUIRED if c not in pred.columns]
    if missing:
        errors.append("missing columns: %s" % missing)
    else:
        if pred["run_id"].duplicated().any():
            errors.append("duplicated run_id (%d)" % int(pred["run_id"].duplicated().sum()))
        bad = sorted(set(pred["label"].astype(str)) - set(LABELS))
        if bad:
            errors.append("unknown labels: %s" % bad[:5])
        need, have = set(test["run_id"]), set(pred["run_id"])
        if need - have:
            errors.append("missing %d run_id from test.csv" % len(need - have))
        if have - need:
            errors.append("%d unknown run_id not present in test.csv" % len(have - need))
        succ = pd.to_numeric(pred["success"], errors="coerce")
        if succ.isna().any() or not set(succ.dropna().unique()).issubset({0, 1}):
            errors.append("success must be 0 or 1")
        ft = pd.to_numeric(pred["fault_turn"], errors="coerce")
        if ft.isna().any():
            errors.append("fault_turn must be an integer (-1 when unknown)")

    if errors:
        for e in errors:
            print("FAIL: %s" % e)
        sys.exit(1)
    print("OK: %d rows, format valid" % len(pred))
    print(pred["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
