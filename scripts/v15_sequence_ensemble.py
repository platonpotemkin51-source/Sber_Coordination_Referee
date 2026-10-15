"""Audit a frozen v15/v9/sequence_v6 probability ensemble.

All component models stay unchanged. Class-wise sequence weights are selected
on the development selection half, then applied unchanged to check-half and
fold 0. Success and fault-turn predictions remain from the v15 export.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import v15_mix_audit as audit


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "handoff/v15/assembled/outputs"
EXPORT = ROOT / "handoff/v15/assembled/evaluation_export"
DEST = ROOT / "outputs/v15-sequence-classgate-candidate"
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def table_slice(slice_, indices):
    result = {}
    for key, value in slice_.items():
        if key in ("truth", "membership"):
            result[key] = value.iloc[indices].reset_index(drop=True)
        else:
            result[key] = value[indices]
    return result


def mix(base, sequence, gates):
    probabilities = (1 - gates[None, :]) * base + gates[None, :] * sequence
    return probabilities / np.maximum(probabilities.sum(1, keepdims=True), 1e-12)


def export(slice_, probabilities, filename):
    labels = np.asarray(audit.LABELS)[probabilities.argmax(1)]
    turns = slice_["candidates"][np.arange(len(labels)), probabilities.argmax(1)]
    frame = pd.DataFrame({
        "run_id": slice_["truth"].run_id,
        "label": labels,
        "success": slice_["success"],
        "fault_turn": turns,
    })
    frame.to_csv(DEST / filename, index=False)
    return audit.score_probability(slice_, probabilities)


def bootstrap_delta(slice_, candidate, baseline, seed, samples=2000):
    rng = np.random.default_rng(seed)
    count = len(candidate)
    deltas = np.empty(samples)
    for i in range(samples):
        indices = rng.integers(0, count, size=count)
        sampled = table_slice(slice_, indices)
        deltas[i] = (
            audit.score_probability(sampled, candidate[indices])["score_proxy"]
            - audit.score_probability(sampled, baseline[indices])["score_proxy"]
        )
    return {
        "median_delta": float(np.median(deltas)),
        "ci95": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
        "share_positive": float(np.mean(deltas > 0)),
        "resamples": samples,
        "unit": "paired row bootstrap; descriptive only",
    }


def main():
    dev = audit.load_slice(
        "development_fold1", None, OUTPUTS / "pair_selector_v15/validation.npz"
    )
    fold0 = audit.load_slice(
        "fold0", None, OUTPUTS / "pair_selector_v15/fold0/validation.npz"
    )
    seq = np.load(OUTPUTS / "sequence_v6/validation.npz")
    seq0 = np.load(OUTPUTS / "sequence_v6/fold0/validation.npz")
    rank = np.load(OUTPUTS / "stack_v9/validation.npz")
    rank0 = np.load(OUTPUTS / "stack_v9/fold0/validation.npz")

    np.testing.assert_array_equal(seq["idx"], dev["membership"].official_row_index)
    np.testing.assert_array_equal(seq0["idx"], fold0["membership"].official_row_index)
    np.testing.assert_array_equal(rank["idx"], dev["membership"].official_row_index)
    np.testing.assert_array_equal(rank0["idx"], fold0["membership"].official_row_index)

    check_truth = pd.read_csv(EXPORT / "check_half_truth.csv")
    check_membership = pd.read_csv(EXPORT / "check_half_membership.csv")
    check_ids = check_membership.official_row_index.to_numpy()
    dev_ids = dev["membership"].official_row_index.to_numpy()
    position = {int(value): i for i, value in enumerate(dev_ids)}
    check_pos = np.asarray([position[int(value)] for value in check_ids])
    check = {
        "truth": check_truth,
        "membership": check_membership,
        **{key: value[check_pos] for key, value in dev.items()
           if key not in ("truth", "membership")},
    }
    check_id_set = set(check_ids.tolist())
    selection_pos = np.asarray([
        i for i, value in enumerate(dev_ids) if int(value) not in check_id_set
    ])
    selection = table_slice(dev, selection_pos)

    # Reconstruct the previously audited v15 + v9 candidate from frozen arrays.
    pair_gates = np.asarray([1, 0.5, 1, 0, 0.5, 0.5, 0.5])
    v15_global = audit.predict(dev, np.full(7, 0.5))
    v15_pair = audit.predict(dev, pair_gates)
    base = 0.25 * v15_global + 0.75 * v15_pair
    v9_gates = np.asarray([0.25, 0.5, 0.75, 1, 1, 1, 0.5])
    v15_v9 = (1 - v9_gates[None, :]) * base + v9_gates[None, :] * rank["label_probabilities"]
    v15_v9 /= np.maximum(v15_v9.sum(1, keepdims=True), 1e-12)

    base0 = 0.25 * audit.predict(fold0, np.full(7, 0.5)) + 0.75 * audit.predict(fold0, pair_gates)
    v15_v90 = (1 - v9_gates[None, :]) * base0 + v9_gates[None, :] * rank0["label_probabilities"]
    v15_v90 /= np.maximum(v15_v90.sum(1, keepdims=True), 1e-12)

    # Low-dimensional coordinate search uses selection-half only.
    gates = np.full(7, 0.5)
    for _ in range(3):
        for label_index in range(7):
            options = []
            for value in GRID:
                trial = gates.copy()
                trial[label_index] = value
                score = audit.score_probability(
                    selection,
                    mix(v15_v9[selection_pos], seq["label_probabilities"][selection_pos], trial),
                )["score_proxy"]
                options.append((score, value))
            gates[label_index] = max(options)[1]

    candidate = mix(v15_v9, seq["label_probabilities"], gates)
    candidate0 = mix(v15_v90, seq0["label_probabilities"], gates)
    check_candidate = candidate[check_pos]
    selection_candidate = candidate[selection_pos]

    DEST.mkdir(parents=True, exist_ok=True)
    metrics = {
        "selection_half": export(selection, selection_candidate, "selection_half.csv"),
        "development_fold1": export(dev, candidate, "development_fold1.csv"),
        "check_half": export(check, check_candidate, "check_half.csv"),
        "fold0": export(fold0, candidate0, "fold0.csv"),
    }
    uncertainty = {
        "candidate_vs_original_v15": bootstrap_delta(
            check, check_candidate, audit.predict(dev, np.full(7, 0.5))[check_pos], 20261013
        ),
        "candidate_vs_previous_v15_v9": bootstrap_delta(
            check, check_candidate, v15_v9[check_pos], 20261014
        ),
    }
    config = {
        "source": "frozen v15 + archived stack_v9 + archived sequence_v6 probabilities",
        "selection_seed": 12,
        "selection_objective": "official local score_proxy on development selection-half",
        "sequence_v6_class_gates": gates.tolist(),
        "v15_pair_class_gates": pair_gates.tolist(),
        "v15_pair_gate_strength": 0.75,
        "v9_class_gates": v9_gates.tolist(),
        "success_source": "original v15 saved predictions",
        "fault_turn_source": "original v15 candidate turns, selected by final label",
        "metrics": metrics,
        "check_half_bootstrap": uncertainty,
    }
    (DEST / "configuration_and_metrics.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
