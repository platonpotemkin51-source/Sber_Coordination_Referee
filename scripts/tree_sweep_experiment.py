"""Isolated LightGBM size sweep with a fixed onset model and group split."""
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from referee.data import fixed_folds, load_data
from referee.model import RunModel
from referee.pair_selector import detector_fusion
from referee.relations import relation_features
from scripts.experiment import report


def metrics(data, idx, probability, success, evidence, candidates, training):
    probability = detector_fusion(probability, evidence)
    labels = probability.argmax(1)
    turns = candidates[np.arange(len(idx)), labels]
    return report(data, idx, labels, (success >= .45).astype(int), turns, training)


def compact(result):
    return {key: round(result[key], 5) for key in
        ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2", "score_proxy")}


def main():
    data = relation_features(load_data("data/train.csv.gz"), "data/train.csv.gz")
    folds = fixed_folds(data)
    training = np.flatnonzero(folds > 1)
    valid = np.flatnonzero(folds == 1)
    control = np.flatnonzero(folds == 0)
    select, check = train_test_split(valid, test_size=.5,
        stratify=data["y"][valid], random_state=12)
    dev_turn = np.load(ROOT / "outputs/multiwindow-ranker/development-single_window.npz")
    ctl_turn = np.load(ROOT / "outputs/multiwindow-ranker/fold0-single_window.npz")
    np.testing.assert_array_equal(valid, dev_turn["idx"])
    np.testing.assert_array_equal(control, ctl_turn["idx"])
    select_mask = np.isin(valid, select)
    check_mask = np.isin(valid, check)
    for trees, leaves in ((350, 31), (650, 31), (350, 63), (650, 63)):
        model = RunModel("struct", trees, leaves).fit(data, training)
        pv, sv = model.predict(data, valid)
        pc, sc = model.predict(data, control)
        dev_full = metrics(data, valid, pv, sv, dev_turn["evidence"],
            dev_turn["candidates"][:, :, 0], training)
        dev_select = metrics(data, valid[select_mask], pv[select_mask], sv[select_mask],
            dev_turn["evidence"][select_mask], dev_turn["candidates"][select_mask, :, 0], training)
        dev_check = metrics(data, valid[check_mask], pv[check_mask], sv[check_mask],
            dev_turn["evidence"][check_mask], dev_turn["candidates"][check_mask, :, 0], training)
        ctl = metrics(data, control, pc, sc, ctl_turn["evidence"],
            ctl_turn["candidates"][:, :, 0], training)
        print({"trees": trees, "leaves": leaves, "full": compact(dev_full),
            "selection": compact(dev_select), "check": compact(dev_check),
            "fold0": compact(ctl)}, flush=True)


if __name__ == "__main__":
    main()
