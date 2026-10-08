"""Train a primary top-two selector using existing, audited v9 OOF caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from referee.data import CODE_HASH, load_data, fixed_folds, concatenate
from referee.relations import RELATIONS_HASH, relation_features
from referee.pair_selector import PAIR_HASH, PairSelector, detector_fusion
from scripts.experiment import report


def load_audited_oof(data, training, excluded, sources, stack_folder):
    certificate = json.loads((stack_folder / "metrics.json").read_text())["oof_certificate"]
    model_hash = hashlib.sha256((ROOT / "referee/model.py").read_bytes()).hexdigest()[:12]
    group_hash = hashlib.sha256("\n".join(data["groups"][training]).encode()).hexdigest()
    expected = {"sources": sources, "features_code_hash": CODE_HASH,
                "relations_hash": RELATIONS_HASH, "model_hash": model_hash,
                "training_groups_sha256": group_hash, "inner_seed": 20261014, "inner_folds": 3}
    if certificate != expected:
        raise ValueError("Existing OOF certificate does not match this training population/code")
    signature = hashlib.sha256(json.dumps(certificate, sort_keys=True).encode()).hexdigest()[:16]
    probabilities = np.zeros((len(training), 7))
    evidence = np.zeros_like(probabilities)
    candidates = np.zeros((len(training), 7), dtype=int)
    assigned = np.zeros(len(training), dtype=int)
    cache_proof = []
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=certificate["inner_seed"])
    for inner, (fit, check) in enumerate(cv.split(data["X"].iloc[training], data["y"][training], data["groups"][training])):
        fitting, checking = training[fit], training[check]
        path = ROOT / "outputs/cache" / f"stack-oof-{signature}-{inner}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Required honest OOF cache is missing: {path}; no detector refit attempted")
        with np.load(path) as cached:
            np.testing.assert_array_equal(cached["fitting"], fitting)
            np.testing.assert_array_equal(cached["checking"], checking)
            assert not set(data["groups"][fitting]) & set(data["groups"][checking])
            assert not set(data["groups"][fitting]) & set(data["groups"][excluded])
            assert not set(data["groups"][checking]) & set(data["groups"][excluded])
            p, e, c = (cached[key] for key in ("probabilities", "evidence", "candidates"))
            if p.shape != (len(check), 7) or e.shape != p.shape or c.shape != p.shape:
                raise ValueError("OOF cache prediction shape mismatch")
            if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
                raise ValueError("Invalid cached run probabilities")
            np.testing.assert_allclose(p.sum(1), 1, rtol=0, atol=1e-8)
            np.testing.assert_array_equal(c[:, 0], -np.ones(len(c), dtype=int))
            lengths = np.diff(data["offsets"])[checking, None]
            if np.any(c[:, 1:] < 0) or np.any(c[:, 1:] >= lengths):
                raise ValueError("Cached onset candidates outside their run")
            detector_fusion(p, e)
            probabilities[check], evidence[check], candidates[check] = p, e, c
        assigned[check] += 1
        cache_proof.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "fitting_rows": len(fitting), "checking_rows": len(checking)})
    if not np.all(assigned == 1):
        raise ValueError("Each meta-training row must have exactly one out-of-fold prediction")
    return probabilities, evidence, candidates, {**certificate, "cache_files": cache_proof, "every_training_row_oof_once": True}


def read_detector(folder, expected_idx):
    metrics = json.loads((folder / "metrics.json").read_text())
    config = metrics["configuration"]
    if metrics.get("features_code_hash") != CODE_HASH or metrics.get("relations_hash") != RELATIONS_HASH:
        raise ValueError("Detector feature provenance mismatch")
    if config.get("turn_weight") != .5 or config.get("turn_power") != 2:
        raise ValueError("Saved predictions were not produced by the expected v5 fusion")
    with np.load(folder / "validation.npz") as saved, np.load(folder / "turns.npz") as turns:
        np.testing.assert_array_equal(saved["idx"], expected_idx)
        fused, evidence, candidates = (saved["label_probabilities"], turns["evidence"], turns["candidates"])
        squared = evidence ** 2
        run_probabilities = 2 * fused - squared / squared.sum(1, keepdims=True)
        if np.any(run_probabilities < -1e-10) or np.any(run_probabilities > 1 + 1e-10):
            raise ValueError("Cannot reconstruct valid run probabilities from saved fusion")
        run_probabilities = np.clip(run_probabilities, 0, 1)
        np.testing.assert_allclose(run_probabilities.sum(1), 1, rtol=0, atol=1e-8)
        np.testing.assert_allclose(detector_fusion(run_probabilities, evidence), fused, rtol=0, atol=1e-10)
        result = {"probabilities": run_probabilities, "evidence": evidence, "candidates": candidates,
                  "original": fused, "success": saved["success_probabilities"]}
    return result, config


def frozen_success(folder, idx, fallback):
    if folder is None:
        return fallback
    with np.load(folder / "validation.npz") as saved:
        np.testing.assert_array_equal(saved["idx"], idx)
        return saved["success_probabilities"].copy()


def predictions(probabilities, success_probabilities, candidates):
    labels = probabilities.argmax(1)
    success = (success_probabilities >= .45).astype(int)
    turns = candidates[np.arange(len(labels)), labels]
    return labels, success, turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pair_selector_v15")
    parser.add_argument("--train", default="data/competition/train.csv")
    parser.add_argument("--extra-train", default="data/train.csv")
    parser.add_argument("--oof-source", default="stack_v9")
    parser.add_argument("--development", default="relations_v5")
    parser.add_argument("--control", default="control_relations_v5")
    parser.add_argument("--nn-success", default="sequence_v6", help="Frozen success probabilities; empty string uses the tree")
    args = parser.parse_args()
    started = time.monotonic()
    data = relation_features(load_data(args.train), args.train)
    folds = fixed_folds(data)
    valid, fold0 = np.flatnonzero(folds == 1), np.flatnonzero(folds == 0)
    training = np.flatnonzero(folds > 1)
    control_training = np.flatnonzero(folds != 0)
    sources = [data["source_sha256"]]
    if args.extra_train:
        extra = relation_features(load_data(args.extra_train), args.extra_train)
        sources.append(extra["source_sha256"])
        excluded_groups = set(data["groups"][folds < 2])
        extra_training = np.array([i for i, group in enumerate(extra["groups"]) if group not in excluded_groups]) + len(data["y"])
        excluded_control = set(data["groups"][folds == 0])
        extra_control = np.array([i for i, group in enumerate(extra["groups"]) if group not in excluded_control]) + len(data["y"])
        training = np.concatenate([training, extra_training])
        control_training = np.concatenate([control_training, extra_control])
        data = concatenate(data, extra)
    p, evidence, candidates, certificate = load_audited_oof(
        data, training, np.concatenate([valid, fold0]), sources, ROOT / "outputs" / args.oof_source,
    )
    print(f"Audited OOF rows={len(training)}; fitting only the binary pair selector", flush=True)
    selector = PairSelector().fit(data, training, p, evidence, candidates)
    print(f"Pair target coverage={selector.eligible_training_rows}/{selector.training_rows}; B share={selector.target_b_share:.4f}", flush=True)
    detector, config = read_detector(ROOT / "outputs" / args.development, valid)
    if config.get("validation_fold") != 1:
        raise ValueError("Development detector must exclude folds 0/1")
    nn_folder = ROOT / "outputs" / args.nn_success if args.nn_success else None
    ps = frozen_success(nn_folder, valid, detector["success"])
    selected = selector.predict(data, valid, detector["probabilities"], detector["evidence"], detector["candidates"])
    select, check = train_test_split(np.arange(len(valid)), test_size=.5, stratify=data["y"][valid], random_state=12)
    choices = []
    selection_trials = []
    for weight in (0, .25, .5, .75, 1):
        probabilities = (1 - weight) * detector["original"] + weight * selected
        labels, success, turns = predictions(probabilities, ps, detector["candidates"])
        selection = report(data, valid[select], labels[select], success[select], turns[select], training)
        selection_trials.append({"selector_weight": weight, "selection_score_proxy": selection["score_proxy"]})
        choices.append((selection["score_proxy"], weight, probabilities, labels, success, turns))
    _, weight, probabilities, labels, success, turns = max(choices, key=lambda result: result[0])
    parameters = {"selector_weight": weight, "success_source": args.nn_success or "tree",
                  "success_threshold": .45, "onset_source": args.development, "fusion": ".5runP+.5norm(evidence^2)"}
    folder = ROOT / "outputs" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf8")
    certificate["pair_hash"] = PAIR_HASH
    certificate["training_rows"] = selector.training_rows
    certificate["eligible_training_rows"] = selector.eligible_training_rows
    certificate["target_b_share"] = selector.target_b_share
    joblib.dump({"selector": selector, "parameters": parameters, "certificate": certificate}, folder / "model.joblib", compress=3)
    metrics = report(data, valid, labels, success, turns, training)
    metrics["check"] = report(data, valid[check], labels[check], success[check], turns[check], training)
    metrics["baseline_with_same_success"] = report(data, valid, *predictions(detector["original"], ps, detector["candidates"]), training)
    metrics.update(configuration={**vars(args), **parameters}, selection_trials=selection_trials, certificate=certificate,
                   elapsed_seconds=time.monotonic() - started, note="Meta and development detector both trained on 16k; epoch/model parameters fixed, only mix selected on selection-half.")
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
    np.savez_compressed(folder / "validation.npz", idx=valid, label_probabilities=probabilities,
                        success_probabilities=ps, labels=labels, success=success, turns=turns,
                        candidates=detector["candidates"], selector_probabilities=selected, folds=folds)
    print(json.dumps(metrics, indent=2), flush=True)
    # Choice is already persisted. Nothing below influences the selected weight.
    control, control_config = read_detector(ROOT / "outputs" / args.control, fold0)
    if control_config.get("validation_fold") != 0:
        raise ValueError("Control detector must exclude fold 0")
    control_ps = frozen_success(nn_folder / "fold0" if nn_folder else None, fold0, control["success"])
    control_selected = selector.predict(data, fold0, control["probabilities"], control["evidence"], control["candidates"])
    control_probabilities = (1 - weight) * control["original"] + weight * control_selected
    control_labels, control_success, control_turns = predictions(control_probabilities, control_ps, control["candidates"])
    control_metrics = report(data, fold0, control_labels, control_success, control_turns, control_training)
    control_metrics["baseline_with_same_success"] = report(data, fold0, *predictions(control["original"], control_ps, control["candidates"]), control_training)
    control_metrics.update(configuration={**vars(args), **parameters}, meta_training_rows=len(training),
                           base_detector_training_rows=len(control_training),
                           note="Frozen pair model/mix on previously opened fold 0; base detector uses 18k, meta selector and neural success use 16k. This is a repeated control, not a fresh independent holdout.")
    target = folder / "fold0"
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(json.dumps(control_metrics, indent=2), encoding="utf8")
    np.savez_compressed(target / "validation.npz", idx=fold0, label_probabilities=control_probabilities,
                        success_probabilities=control_ps, labels=control_labels, success=control_success,
                        turns=control_turns, candidates=control["candidates"], selector_probabilities=control_selected, folds=folds)
    print(json.dumps({"fold0": control_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
