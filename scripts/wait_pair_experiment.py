"""Ablate temporal wait state around the two OOF fault candidates."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "4")
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from baseline.features import parse_run
from referee.data import load_data, concatenate
from referee.relations import relation_features
from referee.pair_selector import pair_features, redistribute_pair
from referee.wait_lifecycle import extract_wait_lifecycle
from wait_lifecycle_experiment import subset
import v15_mix_audit as audit

HANDOFF = ROOT / "handoff/v15/assembled"
DEST = ROOT / "outputs/wait-pair-20261015"


def temporal_features(steps, global_features, indexes, pairs, candidates):
    width = steps[0].shape[1]
    local = np.zeros((len(indexes), 2, width * 4 + 1), dtype=np.float32)
    for row, index in enumerate(indexes):
        series = steps[index]
        for side, label in enumerate(pairs[row]):
            turn = int(candidates[row, label])
            if label and 0 <= turn < len(series):
                local[row, side, 0] = 1
                local[row, side, 1:] = np.concatenate([
                    series[turn], series[-1], series[turn:].mean(0),
                    series[max(0, turn-3):turn+1].mean(0)])
    return np.hstack([global_features.iloc[indexes].to_numpy(),
                      local.reshape(len(indexes), -1), local[:, 0] - local[:, 1]])


def main():
    started = time.monotonic()
    DEST.mkdir(parents=True, exist_ok=True)
    population = pd.read_csv(HANDOFF / "evaluation_export/population_and_truth.csv")
    oof = pd.read_csv(HANDOFF / "evaluation_export/meta_training_oof.csv")
    indexes = oof.global_index.to_numpy(dtype=int)
    assert population.iloc[indexes].meta_and_neural_training.all()
    assert not set(population.iloc[indexes].group_sha256) & set(
        population.loc[~population.meta_and_neural_training, "group_sha256"])
    pieces, steps, global_rows = [], [], []
    for path in (HANDOFF / "data/competition/train.csv", HANDOFF / "data/train.csv"):
        cache = "/private/tmp/v15_oof_feature_cache"
        pieces.append(relation_features(load_data(path, cache=cache), path, cache=cache))
        for record in pd.read_csv(path).to_dict("records"):
            summary, trace = extract_wait_lifecycle(parse_run(record), return_steps=True)
            global_rows.append(summary)
            steps.append(np.asarray([list(row.values()) for row in trace], dtype=np.float32))
    data = concatenate(*pieces)
    del pieces
    np.testing.assert_array_equal(data["run_id"], population.run_id)
    global_features = pd.DataFrame(global_rows).astype(np.float32)
    p = oof[[f"run_p_{name}" for name in audit.LABELS]].to_numpy(dtype=np.float32)
    evidence = oof[[f"evidence_{name}" for name in audit.LABELS]].to_numpy(dtype=np.float32)
    candidates = oof[[f"candidate_{name}" for name in audit.LABELS]].to_numpy(dtype=int)
    x, pairs, _ = pair_features(data, indexes, p, evidence, candidates)
    extra = temporal_features(steps, global_features, indexes, pairs, candidates)
    y = data["y"][indexes]
    eligible = (pairs == y[:, None]).any(1)
    target = (pairs[:, 1] == y).astype(int)
    dev = audit.load_slice("development_fold1", None, HANDOFF / "outputs/pair_selector_v15/validation.npz")
    fold = audit.load_slice("fold0", None, HANDOFF / "outputs/pair_selector_v15/fold0/validation.npz")
    external = []
    for folder, split in (("relations_v5", dev), ("control_relations_v5", fold)):
        saved = np.load(HANDOFF / f"outputs/{folder}/validation.npz")
        turns = np.load(HANDOFF / f"outputs/{folder}/turns.npz")
        ids = split["membership"].official_row_index.to_numpy()
        np.testing.assert_array_equal(saved["idx"], ids)
        ep = turns["evidence"] ** 2
        ep /= np.maximum(ep.sum(1, keepdims=True), 1e-12)
        rp = np.maximum(2*saved["label_probabilities"]-ep, 0)
        rp /= np.maximum(rp.sum(1, keepdims=True), 1e-12)
        xx, pp, fused = pair_features(data, ids, rp, turns["evidence"], turns["candidates"])
        xx_extra = temporal_features(steps, global_features, ids, pp, turns["candidates"])
        external.append((xx, xx_extra, pp, fused))
    check_ids = set(pd.read_csv(HANDOFF / "evaluation_export/check_half_membership.csv").official_row_index)
    ids = dev["membership"].official_row_index.to_numpy()
    cp = np.asarray([i for i, value in enumerate(ids) if value in check_ids])
    sp = np.asarray([i for i, value in enumerate(ids) if value not in check_ids])
    selection, check = subset(dev, sp), subset(dev, cp)
    base = np.load(ROOT / "outputs/char-tfidf-v15-experiment/predictions.npz")
    bd, bf = base["word_char_blend_dev"], base["word_char_blend_fold0"]
    report = {"training_rows": len(indexes), "eligible_pairs": int(eligible.sum()),
              "original_columns": x.shape[1], "added_columns": extra.shape[1],
              "group_overlap": 0, "warning": "Reused development splits; not new independent holdouts."}
    arrays = {}
    for name, temporal in (("original_pair", False), ("temporal_pair", True)):
        features = np.hstack([x, extra]) if temporal else x
        print("Training", name, features.shape, flush=True)
        model = LGBMClassifier(n_estimators=450, learning_rate=.04, num_leaves=15,
            min_child_samples=80, reg_lambda=10, colsample_bytree=.85,
            n_jobs=4, random_state=42, verbosity=-1)
        model.fit(features[eligible], target[eligible])
        predictions = []
        accuracies = []
        for split, (xx, xx_extra, pp, fused) in zip((dev, fold), external):
            xx = np.hstack([xx, xx_extra]) if temporal else xx
            probability_b = model.predict_proba(xx)[:, 1]
            predictions.append(redistribute_pair(fused, pp, probability_b))
            truth = split["truth"].label.map({label: i for i, label in enumerate(audit.LABELS)}).to_numpy()
            mask = (pp == truth[:, None]).any(1)
            accuracies.append(float(np.mean((probability_b[mask] >= .5) == (pp[mask, 1] == truth[mask]))))
        pdv, pf = predictions
        trials = [(audit.score_probability(selection, ((1-a)*bd+a*pdv)[sp])["score_proxy"], -a)
                  for a in (0., .15, .3, .5, .75)]
        alpha = -max(trials)[1]
        qd, qf = (1-alpha)*bd+alpha*pdv, (1-alpha)*bf+alpha*pf
        report[name] = {"alpha": alpha, "selection_trials": [{"alpha": -a, "score": s} for s, a in trials],
            "eligible_pair_accuracy_dev_fold0": accuracies,
            "standalone_check": audit.score_probability(check, pdv[cp]),
            "standalone_fold0": audit.score_probability(fold, pf),
            "blend_check": audit.score_probability(check, qd[cp]),
            "blend_fold0": audit.score_probability(fold, qf)}
        arrays.update({name+"_dev": pdv, name+"_fold0": pf})
        joblib.dump(model, DEST / f"{name}.joblib", compress=3)
        print(name, report[name], flush=True)
    report["seconds"] = time.monotonic() - started
    (DEST / "metrics.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(DEST / "predictions.npz", **arrays)
    print("Saved", DEST, "seconds", report["seconds"], flush=True)


if __name__ == "__main__":
    main()
