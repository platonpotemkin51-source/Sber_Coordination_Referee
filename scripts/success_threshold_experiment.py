"""Selection-only success threshold calibration on a fixed selector blend."""
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from referee.data import fixed_folds, load_data
from referee.model import RunModel
from referee.relations import relation_features
from scripts.experiment import report


def score(data, idx, labels, success, turns, training):
    result = report(data, idx, labels, success, turns, training)
    return result["score_proxy"], result["success_f1"]


def main():
    data = relation_features(load_data("data/train.csv.gz"), "data/train.csv.gz")
    folds = fixed_folds(data)
    training = np.flatnonzero(folds > 1)
    valid, control = np.flatnonzero(folds == 1), np.flatnonzero(folds == 0)
    selection, check = train_test_split(valid, test_size=.5,
        stratify=data["y"][valid], random_state=12)
    selmask, chkmask = np.isin(valid, selection), np.isin(valid, check)
    d = np.load(ROOT / "outputs/multiwindow-ranker/development-single_window.npz")
    t = np.load(ROOT / "outputs/multiwindow-ranker/development-top_three.npz")
    pair = np.load(ROOT / "outputs/success-aware-pair/development.npz")
    p = .5*d["selector_probabilities"] + .25*t["selector_probabilities"] + .25*pair["standard"]
    labels = p.argmax(1)
    turns = d["candidates"][np.arange(len(valid)), labels, 0]
    d0 = np.load(ROOT / "outputs/multiwindow-ranker/fold0-single_window.npz")
    t0 = np.load(ROOT / "outputs/multiwindow-ranker/fold0-top_three.npz")
    pair0 = np.load(ROOT / "outputs/success-aware-pair/fold0.npz")
    p0 = .5*d0["selector_probabilities"] + .25*t0["selector_probabilities"] + .25*pair0["standard"]
    labels0 = p0.argmax(1)
    turns0 = d0["candidates"][np.arange(len(control)), labels0, 0]
    model = RunModel("struct", 350, 31).fit(data, training)
    _, ps = model.predict(data, valid)
    _, ps0 = model.predict(data, control)
    results = []
    for threshold in np.arange(.30, .951, .025):
        s = (ps >= threshold).astype(int)
        selection_score, selection_f1 = score(data, valid[selmask], labels[selmask], s[selmask],
            turns[selmask], training)
        check_score, check_f1 = score(data, valid[chkmask], labels[chkmask], s[chkmask],
            turns[chkmask], training)
        full_score, full_f1 = score(data, valid, labels, s, turns, training)
        results.append((selection_f1, threshold, full_score, full_f1, check_score, check_f1))
    best = max(results)
    threshold = best[1]
    s0 = (ps0 >= threshold).astype(int)
    print({"selection_threshold": threshold, "selection_success_f1": round(best[0], 5),
        "full": {"score": round(best[2],5), "success_f1": round(best[3],5)},
        "check": {"score": round(best[4],5), "success_f1": round(best[5],5)},
        "fold0": {"score": round(score(data,control,labels0,s0,turns0,training)[0],5),
            "success_f1": round(score(data,control,labels0,s0,turns0,training)[1],5)}})
    print("grid", [(round(x[1],3), round(x[0],4), round(x[4],4)) for x in results])


if __name__ == "__main__":
    main()
