"""Inspect paired fold-0 disagreements for deadlock and goal_drift."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "handoff/v15/assembled"
sys.path.insert(0, str(HANDOFF))
from referee.data import load_data

DEST = ROOT / "outputs/char-tfidf-v15-experiment"
LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock", "conflict", "goal_drift", "runaway_loop"]


def main():
    official_path = HANDOFF / "data/competition/train.csv"
    data = load_data(official_path, cache="/private/tmp/v15_oof_feature_cache")
    membership = pd.read_csv(HANDOFF / "evaluation_export/fold0_membership.csv")
    truth = pd.read_csv(HANDOFF / "evaluation_export/fold0_truth.csv")
    ids = membership.official_row_index.to_numpy()
    v15 = np.load(HANDOFF / "outputs/pair_selector_v15/fold0/validation.npz")["label_probabilities"]
    text = np.load(DEST / "predictions.npz")["word_char_blend_fold0"]
    old_i, new_i = v15.argmax(1), text.argmax(1)
    old_y, new_y = np.asarray(LABELS)[old_i], np.asarray(LABELS)[new_i]
    y = truth.label.to_numpy()
    reports = {}
    for cls in ("deadlock", "goal_drift"):
        cases = {
            "v15_right_wordchar_wrong": np.flatnonzero((y == cls) & (old_y == cls) & (new_y != cls)),
            "wordchar_right_v15_wrong": np.flatnonzero((y == cls) & (old_y != cls) & (new_y == cls)),
            "both_wrong_different": np.flatnonzero((y == cls) & (old_y != cls) & (new_y != cls) & (old_y != new_y)),
        }
        reports[cls] = {"counts": {name: int(len(ix)) for name, ix in cases.items()}, "examples": {}}
        for name, positions in cases.items():
            chosen = sorted(positions, key=lambda i: abs(float(v15[i, LABELS.index(cls)] - text[i, LABELS.index(cls)])), reverse=True)[:5]
            rows = []
            for pos in chosen:
                doc = str(data["documents"][int(ids[pos])])
                rows.append({
                    "run_id": str(truth.run_id.iloc[pos]),
                    "truth": str(y[pos]), "v15": str(old_y[pos]), "word_char": str(new_y[pos]),
                    "v15_target_probability": float(v15[pos, LABELS.index(cls)]),
                    "word_char_target_probability": float(text[pos, LABELS.index(cls)]),
                    "document_excerpt": doc[:900],
                })
            reports[cls]["examples"][name] = rows
    output = DEST / "deadlock_goal_drift_examples.json"
    output.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({cls: {"counts": part["counts"],
                            "sample_examples": {k: [{"run_id": r["run_id"], "v15": r["v15"],
                                                    "word_char": r["word_char"],
                                                    "excerpt": r["document_excerpt"][:350]}
                                                   for r in rows[:2]]
                                               for k, rows in part["examples"].items()}}
                     for cls, part in reports.items()}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
