"""Inference wrapper for the audited v15 + v9 class-gated candidate.

The original v15 model files are loaded read-only. The wrapper combines the
existing pair selector and archived v9 ranker with fixed gates selected on the
development selection half. It does not fit any model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PAIR_CLASS_GATE = np.array([1., .5, 1., 0., .5, .5, .5], dtype=np.float32)
V15_GATE_STRENGTH = .75
V9_CLASS_GATE = np.array([.25, .5, .75, 1., 1., 1., .5], dtype=np.float32)


def infer_candidate(bundle, ranker, data, weights_directory, neural_device="auto"):
    from referee.data import LABELS
    from referee.neural_success import success_probabilities

    indexes = np.arange(len(data["run_id"]))
    run_p, tree_success = bundle["run_model"].predict(data, indexes)
    candidates, evidence = bundle["turn_model"].predict(data, indexes)
    evidence_p = evidence ** bundle.get("turn_power", 2)
    evidence_p /= np.maximum(evidence_p.sum(1, keepdims=True), 1e-12)
    base = (1 - bundle.get("turn_weight", .5)) * run_p + bundle.get("turn_weight", .5) * evidence_p

    pair_p = bundle["pair_selector"].predict(data, indexes, run_p, evidence, candidates)
    pair_class = (1 - PAIR_CLASS_GATE[None, :]) * base + PAIR_CLASS_GATE[None, :] * pair_p
    pair_class /= np.maximum(pair_class.sum(1, keepdims=True), 1e-12)
    v15 = (1 - V15_GATE_STRENGTH) * base + V15_GATE_STRENGTH * pair_class

    ranked = ranker.predict(data, indexes, run_p, evidence, candidates)
    v9 = .25 * base + .75 * ranked
    probabilities = (1 - V9_CLASS_GATE[None, :]) * v15 + V9_CLASS_GATE[None, :] * v9
    probabilities /= np.maximum(probabilities.sum(1, keepdims=True), 1e-12)

    if bundle.get("neural_success"):
        success_p = success_probabilities(data, bundle["neural_success"],
            weights_directory, neural_device)
    else:
        success_p = tree_success
    labels = probabilities.argmax(1)
    return pd.DataFrame({
        "run_id": data["run_id"],
        "label": [LABELS[i] for i in labels],
        "success": (success_p >= bundle.get("success_threshold", .45)).astype(int),
        "fault_turn": candidates[np.arange(len(labels)), labels],
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True,
        help="Merged v15 handoff folder containing weights/ and outputs/stack_v9/")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="predictions-v15-classgate.csv")
    parser.add_argument("--neural-device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    root = Path(args.artifact_root).resolve()
    sys.path.insert(0, str(root))
    from referee.data import load_data
    from referee.relations import relation_features

    bundle = joblib.load(root / "weights/model.joblib")
    ranker_bundle = joblib.load(root / "outputs/stack_v9/model.joblib")
    if bundle.get("pair_selector") is None or ranker_bundle.get("ranker") is None:
        raise ValueError("Handoff must contain the frozen v15 pair selector and v9 ranker")
    test = relation_features(load_data(args.test), args.test)
    predictions = infer_candidate(bundle, ranker_bundle["ranker"], test,
        root / "weights", args.neural_device)
    predictions.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(predictions)} rows", flush=True)


if __name__ == "__main__":
    main()
