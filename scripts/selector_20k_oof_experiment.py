"""Cross-fit the pair selector with archived OOF rows plus outer held-out rows."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "handoff/v15/assembled"
sys.path.insert(0, str(HANDOFF))

from referee.data import concatenate, load_data
from referee.pair_selector import PairSelector, detector_fusion
from referee.relations import relation_features


EXPORT = HANDOFF / "evaluation_export"
OUTPUTS = HANDOFF / "outputs"
DEST = ROOT / "outputs/selector-20k-oof-experiment"


def load_data_features():
    cache = "/private/tmp/v15_oof_feature_cache"
    paths = [HANDOFF / "data/competition/train.csv", HANDOFF / "data/train.csv"]
    pieces = [relation_features(load_data(path, cache=cache), path, cache=cache) for path in paths]
    return concatenate(*pieces)


def load_detector(split_name, idx_expected):
    saved = np.load(OUTPUTS / split_name / "validation.npz")
    turns = np.load(OUTPUTS / split_name / "turns.npz")
    np.testing.assert_array_equal(saved["idx"], idx_expected)
    probabilities = np.asarray(saved["label_probabilities"], dtype=np.float64)
    evidence = np.asarray(turns["evidence"], dtype=np.float64)
    candidates = np.asarray(turns["candidates"], dtype=np.int64)
    evidence_p = evidence ** 2
    evidence_p /= np.maximum(evidence_p.sum(1, keepdims=True), 1e-12)
    run_p = 2 * probabilities - evidence_p
    run_p = np.maximum(run_p, 0)
    run_p /= np.maximum(run_p.sum(1, keepdims=True), 1e-12)
    return run_p, evidence, candidates, detector_fusion(run_p, evidence)


def fit_selector(data, meta_idx, meta_p, meta_e, meta_c):
    selector = PairSelector().fit(data, meta_idx, meta_p, meta_e, meta_c)
    return selector


def evaluate_fold(data, selector, name, idx, success, slices, training_size):
    p, evidence, candidates, fused = load_detector(name, idx)
    selected = selector.predict(data, idx, p, evidence, candidates)
    if name != "relations_v5":
        # The weight selected on development is frozen for the separate control.
        weight = slices[4]
        q = (1 - weight) * fused + weight * selected
        q /= np.maximum(q.sum(1, keepdims=True), 1e-12)
        from v15_mix_audit import score_probability
        score = score_probability(slices[0], q)
        return {"weight": weight, "score": score, "probabilities": q,
                "selector": selector, "training_size": training_size}

    w_grid = (0., .25, .5, .75, 1.)
    dev, check, sel_pos, check_pos = slices
    trials = []
    for weight in w_grid:
        q = (1 - weight) * fused + weight * selected
        q /= np.maximum(q.sum(1, keepdims=True), 1e-12)
        q_slice = {k: (v.iloc[sel_pos].reset_index(drop=True)
                       if k in ("truth", "membership") else v[sel_pos])
                   for k, v in dev.items()}
        import v15_mix_audit as audit
        if name == "relations_v5":
            selection_score = audit.score_probability(q_slice, q[sel_pos])["score_proxy"]
            check_score = audit.score_probability(check, q[check_pos])["score_proxy"]
            dev_score = audit.score_probability(dev, q)["score_proxy"]
        else:
            selection_score = None
            check_score = None
            dev_score = None
        trials.append({"weight": weight, "selection": selection_score,
                       "check": check_score, "development": dev_score})
    weight = max(trials, key=lambda item: item["selection"])["weight"]
    q = (1 - weight) * fused + weight * selected
    q /= np.maximum(q.sum(1, keepdims=True), 1e-12)
    return {"trials": trials, "weight": weight, "probabilities": q,
            "selector": selector, "training_size": training_size}


def main():
    import v15_mix_audit as audit

    data = load_data_features()
    population = pd.read_csv(EXPORT / "population_and_truth.csv").set_index("global_index")
    oof = pd.read_csv(EXPORT / "meta_training_oof.csv")
    oof_idx = oof.global_index.to_numpy(dtype=int)
    oof_p = oof[[f"run_p_{name}" for name in audit.LABELS]].to_numpy(dtype=np.float64)
    oof_e = oof[[f"evidence_{name}" for name in audit.LABELS]].to_numpy(dtype=np.float64)
    oof_c = oof[[f"candidate_{name}" for name in audit.LABELS]].to_numpy(dtype=np.int64)
    if len(oof_idx) != 16000 or len(np.unique(oof_idx)) != 16000:
        raise ValueError("Expected the audited 16k OOF training table")

    dev = audit.load_slice("development_fold1", None, OUTPUTS / "pair_selector_v15/validation.npz")
    fold0 = audit.load_slice("fold0", None, OUTPUTS / "pair_selector_v15/fold0/validation.npz")
    dev_idx = dev["membership"].official_row_index.to_numpy()
    fold0_idx = fold0["membership"].official_row_index.to_numpy()
    dev_base = load_detector("relations_v5", dev_idx)
    fold0_base = load_detector("control_relations_v5", fold0_idx)
    np.testing.assert_allclose(dev["base"], dev_base[3], atol=2e-7, rtol=2e-6)
    np.testing.assert_array_equal(dev["candidates"], dev_base[2])
    np.testing.assert_array_equal(fold0["candidates"], fold0_base[2])

    check_ids = pd.read_csv(EXPORT / "check_half_membership.csv").official_row_index.to_numpy()
    positions = {int(value): i for i, value in enumerate(dev_idx)}
    check_pos = np.asarray([positions[int(value)] for value in check_ids])
    check_set = set(check_ids.tolist())
    selection_pos = np.asarray([i for i, value in enumerate(dev_idx) if int(value) not in check_set])
    check_truth = pd.read_csv(EXPORT / "check_half_truth.csv")
    check_membership = pd.read_csv(EXPORT / "check_half_membership.csv")
    check = {"truth": check_truth, "membership": check_membership,
             **{key: value[check_pos] for key, value in dev.items()
                if key not in ("truth", "membership")}}
    slices = (dev, check, selection_pos, check_pos, None)

    # For fold 1, add fold-0 predictions (which were generated by a model that
    # excluded fold 0) to the existing honest OOF meta rows, train, score on
    # selection/check-half. Reverse folds for the control evaluation.
    selector_dev = fit_selector(
        data, np.r_[oof_idx, fold0_idx],
        np.vstack([oof_p, fold0_base[0]]),
        np.vstack([oof_e, fold0_base[1]]),
        np.vstack([oof_c, fold0_base[2]]),
    )
    dev_eval = evaluate_fold(
        data, selector_dev, "relations_v5", dev_idx, dev["success"],
        (dev, check, selection_pos, check_pos), len(oof_idx) + len(fold0_idx),
    )

    selector_fold0 = fit_selector(
        data, np.r_[oof_idx, dev_idx],
        np.vstack([oof_p, dev_base[0]]),
        np.vstack([oof_e, dev_base[1]]),
        np.vstack([oof_c, dev_base[2]]),
    )
    fold0_eval = evaluate_fold(
        data, selector_fold0, "control_relations_v5", fold0_idx,
        fold0["success"], (fold0, None, None, None, dev_eval["weight"]),
        len(oof_idx) + len(dev_idx),
    )

    results = {
        "protocol": "16k honest OOF + the opposite saved outer fold predictions; each evaluation fold excluded from selector training",
        "development_training_rows": len(oof_idx) + len(fold0_idx),
        "control_training_rows": len(oof_idx) + len(dev_idx),
        "development_selector_trials": dev_eval["trials"],
        "selected_weight": dev_eval["weight"],
        "development": audit.score_probability(dev, dev_eval["probabilities"]),
        "check_half": audit.score_probability(check, dev_eval["probabilities"][check_pos]),
        "fold0": fold0_eval["score"],
    }
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    joblib.dump({"development_selector": selector_dev,
                 "fold0_selector": selector_fold0,
                 "selected_weight": dev_eval["weight"]}, DEST / "models.joblib", compress=3)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
