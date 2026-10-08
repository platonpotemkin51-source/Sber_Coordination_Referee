#!/usr/bin/env python3
"""Local cross-validation on train.csv — the honest way to estimate your score.

    python scripts/local_cv.py --train data/train.csv --folds 3

Reports the same four metrics the leaderboard uses.  ``Robustness F1`` is
approximated by the hardest slice available inside train: long runs with rare
topologies (mesh / blackboard), which is the closest proxy to the shifted part
of the test set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "baseline"))
sys.path.insert(0, os.path.join(HERE, "..", "evaluation"))

from features import extract_features, parse_run     # noqa: E402
from localize import localize                        # noqa: E402
from metrics import (binary_f1, composite, f1_per_class, fault_turn_hit_at_k,  # noqa: E402
                     macro_f1)

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="use only N rows (fast smoke)")
    a = ap.parse_args()

    df = pd.read_csv(a.train)
    if a.limit:
        df = df.sample(n=min(a.limit, len(df)), random_state=0).reset_index(drop=True)

    runs = [parse_run(r) for r in df.to_dict("records")]
    X = pd.DataFrame([extract_features(r) for r in runs]).fillna(0.0)
    y = df["label"].values
    s = df["success"].values
    ft = df["fault_turn"].values
    hard = ((X["n_messages"] > X["n_messages"].median())
            & (X["topo_mesh"] + X["topo_blackboard"] > 0)).values

    from sklearn.model_selection import StratifiedKFold
    try:
        import lightgbm as lgb
        mk = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                        num_leaves=63, n_jobs=-1, random_state=42,
                                        verbose=-1)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        mk = lambda: HistGradientBoostingClassifier(max_iter=300, random_state=42)

    oof_lab = np.empty(len(df), dtype=object)
    oof_suc = np.zeros(len(df), dtype=int)
    for tr, te in StratifiedKFold(a.folds, shuffle=True, random_state=0).split(X, y):
        m = mk().fit(X.iloc[tr], y[tr])
        oof_lab[te] = m.predict(X.iloc[te])
        oof_suc[te] = mk().fit(X.iloc[tr], s[tr]).predict(X.iloc[te])

    turns = [localize(r, l) for r, l in zip(runs, oof_lab)]
    res = {
        "macro_f1": macro_f1(list(y), list(oof_lab)),
        "robustness_f1": macro_f1(list(y[hard]), list(oof_lab[hard])) if hard.any() else 0.0,
        "success_f1": binary_f1(list(s), list(oof_suc)),
        "fault_turn_hit2": fault_turn_hit_at_k(list(y), list(oof_lab), list(ft), turns),
    }
    res["score"] = composite(res)
    print(json.dumps(res, indent=2))
    print("\nper-class F1:")
    for c, v in f1_per_class(list(y), list(oof_lab), LABELS).items():
        print("  %-18s %.4f" % (c, v))


if __name__ == "__main__":
    main()
