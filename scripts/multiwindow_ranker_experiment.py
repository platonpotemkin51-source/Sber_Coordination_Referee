"""Pilot a top-three onset-window ranker with honest inner-fold predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from lightgbm import LGBMRanker
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from referee.data import CODE_HASH, LABELS, fixed_folds, load_data
from referee.model import RunModel, TurnModel
from referee.pair_selector import detector_fusion
from referee.relations import RELATIONS_HASH, relation_features
from referee.stack import CandidateRanker
from scripts.experiment import report


def top3_candidates(data, indexes, model):
    """Return up to three non-overlapping onset windows for every fault class."""
    selections = np.concatenate([np.arange(*data["offsets"][i:i + 2]) for i in indexes])
    x = data["turn_X"][selections]
    p = model.clf.predict_proba(x)
    turns = np.full((len(indexes), len(LABELS), 3), -1, dtype=np.int32)
    evidence = np.zeros((len(indexes), len(LABELS)), dtype=np.float32)
    window_scores = np.zeros((len(indexes), len(LABELS), 3), dtype=np.float32)
    cursor = 0
    for row, run_i in enumerate(indexes):
        start, end = data["offsets"][run_i:run_i + 2]
        n = int(end - start)
        local = p[cursor:cursor + n]
        cursor += n
        for cls in range(1, len(LABELS)):
            cls_p = local[:, cls]
            evidence[row, cls] = cls_p.max(initial=0)
            scores = np.asarray([cls_p[max(0, t - 2):min(n, t + 3)].sum() for t in range(n)])
            order = np.argsort(-scores, kind="stable")
            selected = []
            for t in order:
                if all(abs(int(t) - prior) >= 5 for prior in selected):
                    selected.append(int(t))
                    if len(selected) == 3:
                        break
            for rank, t in enumerate(selected):
                turns[row, cls, rank] = t
                window_scores[row, cls, rank] = scores[t]
    evidence[:, 0] = 1 - evidence[:, 1:].max(axis=1)
    return turns, evidence, window_scores


def multiwindow_features(data, indexes, probabilities, evidence, turns, scores):
    n, nclass, top_k = turns.shape
    local_width = data["turn_X"].shape[1]
    local = np.zeros((n, nclass, top_k * local_width * 2), dtype=np.float32)
    positions = np.full((n, nclass, top_k), -1, dtype=np.float32)
    known = np.zeros((n, nclass, top_k), dtype=np.float32)
    for row, run_i in enumerate(indexes):
        start, end = data["offsets"][run_i:run_i + 2]
        length = int(end - start)
        for cls in range(1, nclass):
            for rank, turn in enumerate(turns[row, cls]):
                if 0 <= turn < length:
                    window = data["turn_X"][start + max(0, turn - 2):start + min(length, turn + 3)]
                    offset = rank * local_width * 2
                    local[row, cls, offset:offset + local_width] = window.mean(0)
                    local[row, cls, offset + local_width:offset + 2 * local_width] = window.max(0)
                    positions[row, cls, rank] = turn / max(1, length - 1)
                    known[row, cls, rank] = 1

    fused = detector_fusion(probabilities, evidence).astype(np.float32)
    largest_other_p = np.zeros_like(probabilities)
    largest_other_e = np.zeros_like(evidence)
    for cls in range(nclass):
        other = [j for j in range(nclass) if j != cls]
        largest_other_p[:, cls] = probabilities[:, other].max(1)
        largest_other_e[:, cls] = evidence[:, other].max(1)
    margins = np.column_stack([probabilities, evidence, fused,
        largest_other_p, largest_other_e,
        probabilities - largest_other_p, evidence - largest_other_e,
        fused - np.max(np.where(np.eye(nclass)[None, :, :], -1, fused[:, None, :]), axis=2),
        scores.reshape(n, -1), positions.reshape(n, -1), known.reshape(n, -1),
        -(fused * np.log(np.maximum(fused, 1e-12))).sum(1)])
    globals_ = data["X"].iloc[indexes].to_numpy(dtype=np.float32)
    global_repeated = np.repeat(globals_[:, None, :], nclass, axis=1)
    class_code = np.broadcast_to(np.eye(nclass, dtype=np.float32), (n, nclass, nclass))
    p_repeated = np.repeat(margins[:, None, :], nclass, axis=1)
    return np.concatenate([global_repeated, class_code, p_repeated, local], axis=2).reshape(n * nclass, -1)


class MultiWindowRanker:
    def fit(self, data, indexes, probabilities, evidence, turns, scores):
        self.run_columns = list(data["X"].columns)
        self.turn_columns = list(data["turn_columns"])
        features = multiwindow_features(data, indexes, probabilities, evidence, turns, scores)
        target = (np.arange(len(LABELS))[None, :] == data["y"][indexes, None]).astype(np.int8).ravel()
        frequencies = np.bincount(data["y"][indexes], minlength=len(LABELS))
        weights = len(indexes) / (len(LABELS) * np.maximum(frequencies[data["y"][indexes]], 1))
        self.clf = LGBMRanker(n_estimators=450, learning_rate=.04, num_leaves=31,
            min_child_samples=100, reg_lambda=10, colsample_bytree=.85, n_jobs=8,
            random_state=42, verbosity=-1, lambdarank_truncation_level=2)
        self.clf.fit(features, target, group=np.full(len(indexes), len(LABELS)),
            sample_weight=np.repeat(weights, len(LABELS)))
        return self

    def predict(self, data, indexes, probabilities, evidence, turns, scores):
        features = multiwindow_features(data, indexes, probabilities, evidence, turns, scores)
        raw = self.clf.predict(features).reshape(-1, len(LABELS))
        raw -= raw.max(1, keepdims=True)
        output = np.exp(raw)
        return output / output.sum(1, keepdims=True)


def read_or_fit_oof(data, training, args, destination):
    model_hash = hashlib.sha256((ROOT / "referee/model.py").read_bytes()).hexdigest()[:12]
    candidate_hash = hashlib.sha256(top3_candidates.__code__.co_code).hexdigest()[:12]
    group_hash = hashlib.sha256("\n".join(data["groups"][training]).encode()).hexdigest()[:16]
    signature = f"mw-{data['source_sha256'][:12]}-{CODE_HASH}-{RELATIONS_HASH}-{model_hash}-{candidate_hash}-{group_hash}-{args.trees}"
    probabilities = np.zeros((len(training), 7), dtype=np.float32)
    success = np.zeros(len(training), dtype=np.float32)
    evidence = np.zeros_like(probabilities)
    turns = np.full((len(training), 7, 3), -1, dtype=np.int32)
    scores = np.zeros((len(training), 7, 3), dtype=np.float32)
    assigned = np.zeros(len(training), dtype=np.int8)
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=20261014)
    for inner, (fit, check) in enumerate(cv.split(data["X"].iloc[training], data["y"][training], data["groups"][training])):
        fitting, checking = training[fit], training[check]
        path = destination / "cache" / f"{signature}-{inner}.npz"
        cache_path = path
        if not cache_path.exists():
            compatible = sorted((destination / "cache").glob(
                f"mw-{data['source_sha256'][:12]}-{CODE_HASH}-{RELATIONS_HASH}-{model_hash}-*-{group_hash}-{args.trees}-{inner}.npz"))
            if len(compatible) == 1:
                cache_path = compatible[0]
        if cache_path.exists():
            with np.load(cache_path) as saved:
                np.testing.assert_array_equal(saved["fitting"], fitting)
                np.testing.assert_array_equal(saved["checking"], checking)
                p, ps, e, c, s = (saved[k] for k in ("probabilities", "success", "evidence", "turns", "scores"))
        else:
            print(f"OOF {inner + 1}/3: train={len(fitting)}, held-out={len(checking)}", flush=True)
            model = RunModel("struct", args.trees, 31).fit(data, fitting)
            onset = TurnModel().fit(data, fitting)
            p, ps = model.predict(data, checking)
            c, e, s = top3_candidates(data, checking, onset)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, fitting=fitting, checking=checking,
                probabilities=p, success=ps, evidence=e, turns=c, scores=s)
            del model, onset
        probabilities[check], success[check], evidence[check] = p, ps, e
        turns[check], scores[check] = c, s
        assigned[check] += 1
    if not np.all(assigned == 1):
        raise ValueError("Every meta-training run needs exactly one OOF prediction")
    return probabilities, success, evidence, turns, scores, signature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.csv.gz")
    parser.add_argument("--trees", type=int, default=350)
    parser.add_argument("--validation-fold", type=int, default=1)
    parser.add_argument("--fixed-weight", type=float, default=None,
        help="Use a previously selected mix weight instead of tuning on this validation fold")
    parser.add_argument("--output", default="outputs/multiwindow-ranker")
    args = parser.parse_args()
    start_time = time.monotonic()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    data = relation_features(load_data(args.train), args.train)
    folds = fixed_folds(data)
    valid = np.flatnonzero(folds == args.validation_fold)
    control = np.flatnonzero(folds == 0)
    training_mask = folds != args.validation_fold
    training_mask &= folds != 0
    training = np.flatnonzero(training_mask)
    oof_p, oof_s, oof_e, oof_c, oof_w, signature = read_or_fit_oof(data, training, args, output)
    print("Fitting single-window and top-three rankers on identical OOF predictions", flush=True)
    ranker_one = CandidateRanker().fit(data, training, oof_p, oof_e, oof_c[:, :, 0])
    ranker_three = MultiWindowRanker().fit(data, training, oof_p, oof_e, oof_c, oof_w)

    base = RunModel("struct", args.trees, 31).fit(data, training)
    onset = TurnModel().fit(data, training)
    p_valid, ps_valid = base.predict(data, valid)
    c_valid, e_valid, w_valid = top3_candidates(data, valid, onset)
    p_control, ps_control = base.predict(data, control)
    c_control, e_control, w_control = top3_candidates(data, control, onset)
    fused_valid = detector_fusion(p_valid, e_valid)
    ranked_valid = {
        "single_window": ranker_one.predict(data, valid, p_valid, e_valid, c_valid[:, :, 0]),
        "top_three": ranker_three.predict(data, valid, p_valid, e_valid, c_valid, w_valid),
    }
    success_valid = (ps_valid >= .45).astype(np.int8)
    baseline_labels = fused_valid.argmax(1)
    baseline_turns = c_valid[np.arange(len(valid)), baseline_labels, 0]
    baseline_metrics = report(data, valid, baseline_labels, success_valid, baseline_turns, training)
    if args.fixed_weight is None:
        select, check = train_test_split(np.arange(len(valid)), test_size=.5, stratify=data["y"][valid], random_state=12)
        baseline_metrics["selection_half"] = report(data, valid[select], baseline_labels[select], success_valid[select], baseline_turns[select], training)
        baseline_metrics["check_half"] = report(data, valid[check], baseline_labels[check], success_valid[check], baseline_turns[check], training)
    validation_results = {"baseline_fused": baseline_metrics}
    chosen_by_variant = {}
    for name, ranked in ranked_valid.items():
        trials = []
        mix_weights = (args.fixed_weight,) if args.fixed_weight is not None else (0, .25, .5, .75, 1)
        for mix_weight in mix_weights:
            probability = (1 - mix_weight) * fused_valid + mix_weight * ranked
            labels = probability.argmax(1)
            turns = c_valid[np.arange(len(valid)), labels, 0]
            selected_score = report(data, valid[select], labels[select], success_valid[select], turns[select], training)["score_proxy"] if args.fixed_weight is None else float("nan")
            trials.append((selected_score, mix_weight, probability, labels, turns))
        _, mix_weight, chosen, labels, turns = trials[0] if args.fixed_weight is not None else max(trials, key=lambda x: x[0])
        metrics = report(data, valid, labels, success_valid, turns, training)
        if args.fixed_weight is None:
            metrics["selection_half"] = report(data, valid[select], labels[select], success_valid[select], turns[select], training)
            metrics["check_half"] = report(data, valid[check], labels[check], success_valid[check], turns[check], training)
        else:
            metrics["fixed_weight_source"] = "previous development fold selection; not tuned on this validation fold"
        metrics["baseline_same_success"] = baseline_metrics
        metrics.update(configuration={**vars(args), "ranker_weight": mix_weight},
            selection_trials=[{"weight": w, "score": s if np.isfinite(s) else None} for s, w, *_ in trials],
            oof_cache_signature=signature, elapsed_seconds=time.monotonic() - start_time,
            limitation="10k source only; no v15 checkpoint or second training source available")
        validation_results[name] = metrics
        chosen_by_variant[name] = {"probabilities": chosen, "labels": labels, "turns": turns, "weight": mix_weight}
        np.savez_compressed(output / f"development-{name}.npz", idx=valid, labels=labels, probabilities=chosen,
            raw_probabilities=p_valid, evidence=e_valid, base_probabilities=fused_valid,
            selector_probabilities=ranked, success=success_valid, turns=turns,
            candidates=c_valid, candidate_scores=w_valid, folds=folds)
    validation_results["configuration"] = {**vars(args), "oof_cache_signature": signature,
        "elapsed_seconds": time.monotonic() - start_time}
    (output / "development.json").write_text(json.dumps(validation_results, indent=2), encoding="utf-8")
    print(json.dumps(validation_results, indent=2), flush=True)

    fused_control = detector_fusion(p_control, e_control)
    ranked_control = {
        "single_window": ranker_one.predict(data, control, p_control, e_control, c_control[:, :, 0]),
        "top_three": ranker_three.predict(data, control, p_control, e_control, c_control, w_control),
    }
    control_success = (ps_control >= .45).astype(np.int8)
    control_results = {"baseline_fused": report(data, control, fused_control.argmax(1), control_success,
        c_control[np.arange(len(control)), fused_control.argmax(1), 0], training)}
    for name, ranked in ranked_control.items():
        mix_weight = chosen_by_variant[name]["weight"]
        probabilities = (1 - mix_weight) * fused_control + mix_weight * ranked
        labels = probabilities.argmax(1)
        turns = c_control[np.arange(len(control)), labels, 0]
        metrics = report(data, control, labels, control_success, turns, training)
        metrics["baseline_same_success"] = control_results["baseline_fused"]
        metrics.update(configuration={**vars(args), "ranker_weight": mix_weight},
            note="Repeated fold 0; mix selected on development, not a fresh independent holdout")
        control_results[name] = metrics
        np.savez_compressed(output / f"fold0-{name}.npz", idx=control, labels=labels, probabilities=probabilities,
            raw_probabilities=p_control, evidence=e_control, base_probabilities=fused_control,
            selector_probabilities=ranked, success=control_success, turns=turns,
            candidates=c_control, candidate_scores=w_control, folds=folds)
    (output / "fold0.json").write_text(json.dumps(control_results, indent=2), encoding="utf-8")
    print(json.dumps({"fold0": control_results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
