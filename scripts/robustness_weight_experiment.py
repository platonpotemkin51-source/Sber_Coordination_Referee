"""Targeted training weights for the scorer's observed robustness cohort."""
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from referee.data import fixed_folds, load_data
from referee.model import RunModel, booster
from referee.pair_selector import detector_fusion
from referee.relations import relation_features
from scripts.experiment import report


def compact(data, idx, p, success, evidence, candidates, training):
    p = detector_fusion(p, evidence)
    labels = p.argmax(1)
    turns = candidates[np.arange(len(idx)), labels]
    scores = report(data, idx, labels, (success >= .45).astype(int), turns, training)
    return {key: round(scores[key], 5) for key in
        ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2", "score_proxy")}


def main():
    data = relation_features(load_data("data/train.csv.gz"), "data/train.csv.gz")
    folds = fixed_folds(data)
    training = np.flatnonzero(folds > 1)
    valid = np.flatnonzero(folds == 1)
    select, check = train_test_split(valid, test_size=.5,
        stratify=data["y"][valid], random_state=12)
    select_mask, check_mask = np.isin(valid, select), np.isin(valid, check)
    control = np.flatnonzero(folds == 0)
    dev = np.load(ROOT / "outputs/multiwindow-ranker/development-single_window.npz")
    ctl = np.load(ROOT / "outputs/multiwindow-ranker/fold0-single_window.npz")
    base = RunModel("struct", 350, 31).fit(data, training)
    _, success_dev = base.predict(data, valid)
    _, success_ctl = base.predict(data, control)
    x_train = base.matrix(data, training)
    x_dev = base.matrix(data, valid)
    x_ctl = base.matrix(data, control)
    cutoff = float(data["X"].n_messages.iloc[training].median())
    observed = (data["X"].n_messages.to_numpy() > cutoff) & (
        data["X"].topo_mesh.to_numpy() + data["X"].topo_blackboard.to_numpy() > 0)
    for weight in (1., 1.25, 1.5, 2.):
        sample_weight = np.where(observed[training], weight, 1.).astype(np.float32)
        model = booster(350, 31).fit(x_train, data["y"][training],
            sample_weight=sample_weight)
        p_dev = model.predict_proba(x_dev)
        p_ctl = model.predict_proba(x_ctl)
        print({"hard_weight": weight,
            "full": compact(data, valid, p_dev, success_dev, dev["evidence"],
                dev["candidates"][:, :, 0], training),
            "selection": compact(data, valid[select_mask], p_dev[select_mask],
                success_dev[select_mask], dev["evidence"][select_mask],
                dev["candidates"][select_mask, :, 0], training),
            "check": compact(data, valid[check_mask], p_dev[check_mask],
                success_dev[check_mask], dev["evidence"][check_mask],
                dev["candidates"][check_mask, :, 0], training),
            "fold0": compact(data, control, p_ctl, success_ctl, ctl["evidence"],
                ctl["candidates"][:, :, 0], training)}, flush=True)


if __name__ == "__main__":
    main()
