"""Compare class-level fold-0 errors for colleague, sequence, and word+char."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import v15_mix_audit as audit
from evaluation.metrics import f1_per_class

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "handoff/v15/assembled"
DEST = ROOT / "outputs/char-tfidf-v15-experiment"


def main():
    fold = audit.load_slice("fold0", None,
        HANDOFF / "outputs/pair_selector_v15/fold0/validation.npz")
    original = np.load(HANDOFF / "outputs/pair_selector_v15/fold0/validation.npz")["label_probabilities"]
    predictions = np.load(DEST / "predictions.npz")
    sequence = predictions["baseline_fold0"]
    word_char = predictions["word_char_blend_fold0"]
    y = fold["truth"].label.to_numpy()
    hard = fold["membership"].robustness_proxy.to_numpy(bool)
    labels = audit.LABELS
    matrix = {}
    for name, q in (("colleague_v15", original), ("sequence_ensemble", sequence),
                    ("word_char", word_char)):
        pred = np.asarray(labels)[q.argmax(1)]
        overall = f1_per_class(y, pred, labels)
        robust = f1_per_class(y[hard], pred[hard], labels)
        matrix[name] = {"score": audit.score_probability(fold, q),
                        "per_class_f1": {label: float(overall[label]) for label in labels},
                        "robust_per_class_f1": {label: float(robust[label]) for label in labels}}
    result = {"classes": labels, "rows": len(y), "robustness_rows": int(hard.sum()),
              "models": matrix}
    (DEST / "fold0_class_diagnosis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
