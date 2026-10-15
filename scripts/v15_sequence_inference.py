"""Inference wrapper for the frozen v15 + v9 + sequence_v6 candidate.

The models are loaded read-only. The sequence checkpoint contributes only its
class probabilities; success and fault-turn outputs remain from v15.
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
SEQUENCE_CLASS_GATE = np.array([.5, 0., 1., .5, .25, .5, .5], dtype=np.float32)
WORD_CHAR_CLASS_GATE = np.array([.1, .1, .5, .3, 0., 0., .2], dtype=np.float32)


def predict_sequence(data, checkpoint_path, device="cpu"):
    import torch
    torch.set_num_threads(4)
    from referee.data import CODE_HASH
    from referee.relations import RELATIONS_HASH
    from referee.sequence import SequenceNetwork, predict, signed_log

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["features_code_hash"] != CODE_HASH or checkpoint["relations_hash"] != RELATIONS_HASH:
        raise ValueError("sequence_v6 feature provenance mismatch")

    stats = checkpoint["normalization"]
    run_raw = signed_log(data["X"][checkpoint["run_columns"]].to_numpy(dtype=np.float32))
    run = np.clip((run_raw - stats["run_mean"]) / stats["run_std"], -8, 8)
    turn_columns = [data["turn_columns"].index(column) for column in checkpoint["turn_columns"]]
    turn_raw = signed_log(data["turn_X"][:, turn_columns])
    turns = np.clip((turn_raw - stats["turn_mean"]) / stats["turn_std"], -8, 8)

    network = SequenceNetwork(run.shape[1], turns.shape[1], checkpoint["width"])
    network.load_state_dict(checkpoint["state"])
    network.to(device)
    active = np.flatnonzero(np.diff(data["offsets"]) > 0)
    probabilities = np.zeros((len(data["run_id"]), 7), dtype=np.float32)
    if len(active):
        probabilities[active], _, _ = predict(network, run, turns, data["offsets"], active)
    return probabilities


def infer_candidate(bundle, ranker, data, weights_directory, sequence_checkpoint,
                    neural_device="cpu", text_model_path=None):
    from referee.data import LABELS
    from referee.neural_success import success_probabilities

    indexes = np.arange(len(data["run_id"]))
    run_p, tree_success = bundle["run_model"].predict(data, indexes)
    candidates, evidence = bundle["turn_model"].predict(data, indexes)
    evidence_p = evidence ** bundle.get("turn_power", 2)
    evidence_p /= np.maximum(evidence_p.sum(1, keepdims=True), 1e-12)
    base = ((1 - bundle.get("turn_weight", .5)) * run_p
            + bundle.get("turn_weight", .5) * evidence_p)

    pair_p = bundle["pair_selector"].predict(data, indexes, run_p, evidence, candidates)
    pair_class = (1 - PAIR_CLASS_GATE[None, :]) * base + PAIR_CLASS_GATE[None, :] * pair_p
    pair_class /= np.maximum(pair_class.sum(1, keepdims=True), 1e-12)
    v15 = (1 - V15_GATE_STRENGTH) * base + V15_GATE_STRENGTH * pair_class

    ranked = ranker.predict(data, indexes, run_p, evidence, candidates)
    v9 = .25 * base + .75 * ranked
    v15_v9 = (1 - V9_CLASS_GATE[None, :]) * v15 + V9_CLASS_GATE[None, :] * v9
    v15_v9 /= np.maximum(v15_v9.sum(1, keepdims=True), 1e-12)

    sequence = predict_sequence(data, sequence_checkpoint, neural_device)
    probabilities = ((1 - SEQUENCE_CLASS_GATE[None, :]) * v15_v9
                     + SEQUENCE_CLASS_GATE[None, :] * sequence)
    probabilities /= np.maximum(probabilities.sum(1, keepdims=True), 1e-12)
    has_turns = np.diff(data["offsets"]) > 0
    probabilities[~has_turns] = v15_v9[~has_turns]

    if text_model_path is not None:
        text_bundle = joblib.load(text_model_path)
        word_features = text_bundle["word_vectorizer"].transform(data["documents"])
        char_features = text_bundle["char_vectorizer"].transform(data["documents"])
        from scipy.sparse import hstack
        text_features = hstack([word_features, char_features], format="csr")
        text_model = text_bundle["models"]["word_char"]
        text_probability = text_model.predict_proba(text_features)
        if not np.array_equal(text_model.classes_, np.arange(probabilities.shape[1])):
            raise ValueError("word+char model class order does not match the referee labels")
        probabilities = ((1 - WORD_CHAR_CLASS_GATE[None, :]) * probabilities
                         + WORD_CHAR_CLASS_GATE[None, :] * text_probability)
        probabilities /= np.maximum(probabilities.sum(1, keepdims=True), 1e-12)

    if bundle.get("neural_success"):
        success_p = success_probabilities(data, bundle["neural_success"],
            weights_directory, neural_device)
        success_p[~has_turns] = tree_success[~has_turns]
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
        help="Merged v15 handoff folder with weights/, outputs/, and referee/ code")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="predictions-v15-sequence-ensemble.csv")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--text-model", default=None,
        help="Optional outputs/char-tfidf-v15-experiment/model.joblib word+char model")
    args = parser.parse_args()
    root = Path(args.artifact_root).resolve()
    sys.path.insert(0, str(root))
    from referee.data import load_data
    from referee.relations import relation_features

    bundle = joblib.load(root / "weights/model.joblib")
    ranker_bundle = joblib.load(root / "outputs/stack_v9/model.joblib")
    if bundle.get("pair_selector") is None or ranker_bundle.get("ranker") is None:
        raise ValueError("Handoff is missing the frozen v15 selector or v9 ranker")
    test = relation_features(load_data(args.test), args.test)
    predictions = infer_candidate(
        bundle, ranker_bundle["ranker"], test, root / "weights",
        root / "outputs/sequence_v6/model.pt", args.device, args.text_model,
    )
    predictions.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(predictions)} rows", flush=True)


if __name__ == "__main__":
    main()
