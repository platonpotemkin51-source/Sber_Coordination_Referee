"""Paired row-bootstrap uncertainty for the word/char ensemble gain."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import v15_mix_audit as audit

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "handoff/v15/assembled"
EXPORT = HANDOFF / "evaluation_export"
DEST = ROOT / "outputs/char-tfidf-v15-experiment"


def main():
    dev = audit.load_slice("development_fold1", None,
        HANDOFF / "outputs/pair_selector_v15/validation.npz")
    pred = np.load(DEST / "predictions.npz")
    dev_ids = dev["membership"].official_row_index.to_numpy()
    check_ids = pd.read_csv(EXPORT / "check_half_membership.csv").official_row_index.to_numpy()
    check_truth = pd.read_csv(EXPORT / "check_half_truth.csv")
    position = {int(row): i for i, row in enumerate(dev_ids)}
    ix = np.asarray([position[int(row)] for row in check_ids])
    check = {"truth": check_truth,
             "membership": pd.read_csv(EXPORT / "check_half_membership.csv"),
             "base": dev["base"][ix], "selector": dev["selector"][ix],
             "success": dev["success"][ix], "candidates": dev["candidates"][ix]}
    current, candidate = pred["baseline_dev"][ix], pred["word_char_blend_dev"][ix]
    n = len(ix)
    rng = np.random.default_rng(20261013)
    delta = []
    for _ in range(2000):
        sample = rng.integers(0, n, size=n)
        slice_sample = {key: (value.iloc[sample].reset_index(drop=True) if key in ("truth", "membership") else value[sample])
                        for key, value in check.items()}
        delta.append(audit.score_probability(slice_sample, candidate[sample])["score_proxy"] -
                     audit.score_probability(slice_sample, current[sample])["score_proxy"])
    delta = np.asarray(delta)
    result = {"bootstrap_reps": len(delta), "seed": 20261013,
              "median_delta": float(np.median(delta)),
              "ci_2_5_97_5": np.quantile(delta, [.025, .975]).tolist(),
              "positive_fraction": float(np.mean(delta > 0)),
              "mean_delta": float(np.mean(delta))}
    (DEST / "paired_bootstrap.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
