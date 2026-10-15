"""Fixed ablation: original run features vs additional temporal wait features.

No full-data detector is used. Both branches train on the same 16k rows,
excluding both evaluation folds and their groups. Selection chooses one scalar
blend weight; previously used check/fold0 remain development diagnostics.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "4")
import hashlib
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
from referee.data import load_data
from referee.relations import relation_features
from referee.wait_lifecycle import extract_wait_lifecycle
import v15_mix_audit as audit

HANDOFF = ROOT / "handoff/v15/assembled"
DEST = ROOT / "outputs/wait-lifecycle-20261015"


def subset(split, indexes):
    return {key: value.iloc[indexes].reset_index(drop=True) if key in ("truth", "membership")
            else value[indexes] for key, value in split.items()}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    population = pd.read_csv(HANDOFF / "evaluation_export/population_and_truth.csv")
    training = np.flatnonzero(population.meta_and_neural_training.to_numpy(bool))
    train_groups = set(population.iloc[training].group_sha256)
    assert not train_groups & set(population.loc[~population.meta_and_neural_training, "group_sha256"])
    old, new = [], []
    for path in (HANDOFF / "data/competition/train.csv", HANDOFF / "data/train.csv"):
        cache = "/private/tmp/v15_oof_feature_cache"
        data = relation_features(load_data(path, cache=cache), path, cache=cache)
        old.append(data["X"])
        del data
        frame = pd.read_csv(path)
        rows = []
        for i, record in enumerate(frame.to_dict("records")):
            rows.append(extract_wait_lifecycle(parse_run(record)))
            if i % 2000 == 0:
                print(f"lifecycle {path.parent.name}: {i}/{len(frame)}", flush=True)
        new.append(pd.DataFrame(rows).astype(np.float32))
    old, new = pd.concat(old, ignore_index=True), pd.concat(new, ignore_index=True)
    assert not set(old.columns) & set(new.columns)
    assert np.isfinite(new.to_numpy()).all()
    new.to_csv(DEST / "features.csv", index=False)
    y = population.label.map({name: i for i, name in enumerate(audit.LABELS)}).to_numpy()
    dev = audit.load_slice("development_fold1", None, HANDOFF / "outputs/pair_selector_v15/validation.npz")
    fold = audit.load_slice("fold0", None, HANDOFF / "outputs/pair_selector_v15/fold0/validation.npz")
    did, fid = dev["membership"].official_row_index.to_numpy(), fold["membership"].official_row_index.to_numpy()
    check_ids = set(pd.read_csv(HANDOFF / "evaluation_export/check_half_membership.csv").official_row_index)
    cp = np.asarray([i for i, value in enumerate(did) if value in check_ids])
    sp = np.asarray([i for i, value in enumerate(did) if value not in check_ids])
    selection, check = subset(dev, sp), subset(dev, cp)
    saved = np.load(ROOT / "outputs/char-tfidf-v15-experiment/predictions.npz")
    baseline_dev, baseline_fold = saved["word_char_blend_dev"], saved["word_char_blend_fold0"]
    report = {"training_rows": len(training), "new_features": new.columns.tolist(),
              "group_overlap": 0, "formula": "0.50 macro + 0.25 robustness + 0.15 success + 0.10 hit2",
              "warning": "Previously reused development slices, not independent unseen holdout.",
              "baseline": {"check": audit.score_probability(check, baseline_dev[cp]),
                           "fold0": audit.score_probability(fold, baseline_fold)}}
    arrays = {}
    for name, features in (("original", old), ("with_lifecycle", pd.concat([old, new], axis=1))):
        print(f"Training {name}: {features.shape}, train={len(training)}", flush=True)
        model = LGBMClassifier(n_estimators=350, learning_rate=.04, num_leaves=31,
            min_child_samples=40, reg_lambda=5, colsample_bytree=.85, subsample=.9,
            subsample_freq=1, n_jobs=4, random_state=20261015, verbosity=-1)
        model.fit(features.iloc[training], y[training])
        pdv, pf = model.predict_proba(features.iloc[did]), model.predict_proba(features.iloc[fid])
        trials = []
        for alpha in (0., .15, .3, .5):
            qd = (1-alpha)*baseline_dev + alpha*pdv
            trials.append((audit.score_probability(selection, qd[sp])["score_proxy"], -alpha))
        alpha = -max(trials)[1]
        qd, qf = (1-alpha)*baseline_dev+alpha*pdv, (1-alpha)*baseline_fold+alpha*pf
        report[name] = {"alpha_selected": alpha,
            "selection_trials": [{"alpha": -a, "score": s} for s, a in trials],
            "standalone_check": audit.score_probability(check, pdv[cp]),
            "standalone_fold0": audit.score_probability(fold, pf),
            "blend_check": audit.score_probability(check, qd[cp]),
            "blend_fold0": audit.score_probability(fold, qf),
            "feature_importance": sorted(zip(features.columns, model.feature_importances_.tolist()),
                                         key=lambda item: -item[1])[:40]}
        arrays.update({f"{name}_dev": pdv, f"{name}_fold0": pf,
                       f"{name}_blend_dev": qd, f"{name}_blend_fold0": qf})
        joblib.dump(model, DEST / f"{name}.joblib", compress=3)
        print(name, "alpha", alpha, "check", report[name]["blend_check"]["score_proxy"],
              "fold0", report[name]["blend_fold0"]["score_proxy"], flush=True)
    report["seconds"] = time.monotonic() - started
    report["extractor_sha256"] = hashlib.sha256((ROOT / "referee/wait_lifecycle.py").read_bytes()).hexdigest()
    (DEST / "metrics.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(DEST / "predictions.npz", **arrays)
    print("Saved", DEST, "seconds", report["seconds"], flush=True)


if __name__ == "__main__":
    main()
