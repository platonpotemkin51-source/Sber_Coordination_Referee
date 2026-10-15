"""Small predeclared probability blend study for saved selector outputs."""
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from referee.data import fixed_folds, load_data
from scripts.experiment import report


def evaluate(data, idx, probabilities, success, candidates, training):
    labels = probabilities.argmax(1)
    turns = candidates[np.arange(len(idx)), labels]
    return report(data, idx, labels, success, turns, training)["score_proxy"]


def components(data, idx, probabilities, success, candidates, training):
    labels = probabilities.argmax(1)
    turns = candidates[np.arange(len(idx)), labels]
    result = report(data, idx, labels, success, turns, training)
    return {key: result[key] for key in ("macro_f1", "robustness_f1", "success_f1", "fault_turn_hit2", "score_proxy")}


def main():
    data = load_data("data/train.csv.gz")
    folds = fixed_folds(data)
    dev = np.load(ROOT / "outputs/multiwindow-ranker/development-single_window.npz")
    top = np.load(ROOT / "outputs/multiwindow-ranker/development-top_three.npz")
    pair = np.load(ROOT / "outputs/success-aware-pair/development.npz")
    idx = dev["idx"]
    np.testing.assert_array_equal(idx, top["idx"])
    np.testing.assert_array_equal(idx, pair["idx"])
    train = np.flatnonzero((folds != 0) & (folds != 1))
    selection, check = train_test_split(np.arange(len(idx)), test_size=.5,
        stratify=data["y"][idx], random_state=12)
    sources = [dev["selector_probabilities"], top["selector_probabilities"], pair["standard"]]
    # Coarse simplex grid only; selection half chooses the blend.
    candidates = []
    for a in (0, .25, .5, .75, 1):
        for b in (0, .25, .5, .75, 1-a):
            c = round(1-a-b, 2)
            if c < 0 or c > 1 or abs((a+b+c)-1) > 1e-7:
                continue
            if not any(np.allclose(x, (a,b,c)) for x in candidates):
                candidates.append((a,b,c))
    scored = []
    for weights in candidates:
        p = sum(w*q for w,q in zip(weights, sources))
        scored.append((evaluate(data, idx[selection], p[selection], dev["success"][selection],
            dev["candidates"][selection,:,0], train), weights))
    best_score, best = max(scored, key=lambda x:x[0])
    p = sum(w*q for w,q in zip(best, sources))
    current = sources[0]
    print("best_weights", best, "selection", round(best_score,6))
    print("development_full", round(evaluate(data, idx, p, dev["success"], dev["candidates"][:,:,0], train),6))
    print("development_check", round(evaluate(data, idx[check], p[check], dev["success"][check], dev["candidates"][check,:,0], train),6))
    print("development_components", {k:round(v,6) for k,v in components(data,idx,p,dev["success"],dev["candidates"][:,:,0],train).items()})
    print("check_components", {k:round(v,6) for k,v in components(data,idx[check],p[check],dev["success"][check],dev["candidates"][check,:,0],train).items()})
    print("ranker_only_check", round(evaluate(data, idx[check], current[check], dev["success"][check], dev["candidates"][check,:,0], train),6))
    # Fold-zero is a repeat control; do not retune on it.
    d0 = np.load(ROOT / "outputs/multiwindow-ranker/fold0-single_window.npz")
    t0 = np.load(ROOT / "outputs/multiwindow-ranker/fold0-top_three.npz")
    p0f = np.load(ROOT / "outputs/success-aware-pair/fold0.npz")
    idx0 = d0["idx"]
    p0 = sum(w*q for w,q in zip(best, [d0["selector_probabilities"], t0["selector_probabilities"], p0f["standard"]]))
    tr0 = np.flatnonzero(folds != 0)
    print("fold0_selected", round(evaluate(data, idx0, p0, d0["success"], d0["candidates"][:,:,0], tr0),6))
    print("fold0_single", round(evaluate(data, idx0, d0["selector_probabilities"], d0["success"], d0["candidates"][:,:,0], tr0),6))

    # Calibrate per-class logit offsets on the first half, then freeze them.
    base = dev["probabilities"]
    offsets = np.zeros(base.shape[1])
    def adjusted(offset, rows):
        z = np.log(np.maximum(base[rows], 1e-9)) + offset
        z -= z.max(1, keepdims=True)
        q = np.exp(z)
        return q / q.sum(1, keepdims=True)
    for _ in range(2):
        for cls in range(base.shape[1]):
            choices = []
            for value in np.arange(-.30, .301, .05):
                trial = offsets.copy(); trial[cls] = value
                q = adjusted(trial, selection)
                choices.append((evaluate(data, idx[selection], q, dev["success"][selection],
                    dev["candidates"][selection,:,0], train), value))
            offsets[cls] = max(choices)[1]
    qdev = adjusted(offsets, np.arange(len(idx)))
    print("offsets", np.round(offsets,2).tolist())
    print("offsets_selection", round(evaluate(data, idx[selection], qdev[selection], dev["success"][selection], dev["candidates"][selection,:,0], train),6))
    print("offsets_check", round(evaluate(data, idx[check], qdev[check], dev["success"][check], dev["candidates"][check,:,0], train),6))
    q0z = np.log(np.maximum(d0["probabilities"], 1e-9)) + offsets
    q0z -= q0z.max(1, keepdims=True); q0 = np.exp(q0z); q0 /= q0.sum(1, keepdims=True)
    print("offsets_fold0", round(evaluate(data, idx0, q0, d0["success"], d0["candidates"][:,:,0], tr0),6))

    # Include the detector fusion baseline as an anchor so selector-only
    # mixtures cannot become overconfident.
    four = [dev["base_probabilities"], dev["selector_probabilities"],
        top["selector_probabilities"], pair["standard"]]
    grid4 = [(a,b,c,1-a-b-c) for a in (0,.25,.5,.75,1)
        for b in (0,.25,.5,.75,1-a) for c in (0,.25,.5,.75,1-a-b)
        if a+b+c <= 1+1e-8]
    scored4 = []
    for weights in grid4:
        if min(weights) < -1e-8 or abs(sum(weights)-1) > 1e-7:
            continue
        q = sum(w*x for w,x in zip(weights,four))
        scored4.append((evaluate(data,idx[selection],q[selection],dev["success"][selection],
            dev["candidates"][selection,:,0],train),weights,q))
    best4 = max(scored4,key=lambda x:x[0])
    q = best4[2]
    print("anchored_weights",best4[1],"selection",round(best4[0],6),
        "full",round(evaluate(data,idx,q,dev["success"],dev["candidates"][:,:,0],train),6),
        "check",round(evaluate(data,idx[check],q[check],dev["success"][check],dev["candidates"][check,:,0],train),6))
    four0=[d0["base_probabilities"],d0["selector_probabilities"],t0["selector_probabilities"],p0f["standard"]]
    q0=sum(w*x for w,x in zip(best4[1],four0))
    print("anchored_fold0",round(evaluate(data,idx0,q0,d0["success"],d0["candidates"][:,:,0],tr0),6))
    bias4 = np.zeros(q.shape[1])
    for _ in range(2):
        for cls in range(len(bias4)):
            options=[]
            for value in np.arange(-.25,.251,.05):
                trial=bias4.copy(); trial[cls]=value
                z=np.log(np.maximum(q[selection],1e-9))+trial; z-=z.max(1,keepdims=True)
                z=np.exp(z); z/=z.sum(1,keepdims=True)
                options.append((evaluate(data,idx[selection],z,dev["success"][selection],
                    dev["candidates"][selection,:,0],train),value))
            bias4[cls]=max(options)[1]
    z=np.log(np.maximum(q,1e-9))+bias4; z-=z.max(1,keepdims=True); z=np.exp(z); z/=z.sum(1,keepdims=True)
    print("anchored_bias",np.round(bias4,2).tolist(),"selection",round(evaluate(data,idx[selection],z[selection],dev["success"][selection],dev["candidates"][selection,:,0],train),6),
        "full",round(evaluate(data,idx,z,dev["success"],dev["candidates"][:,:,0],train),6),
        "check",round(evaluate(data,idx[check],z[check],dev["success"][check],dev["candidates"][check,:,0],train),6))
    z0=np.log(np.maximum(q0,1e-9))+bias4; z0-=z0.max(1,keepdims=True); z0=np.exp(z0); z0/=z0.sum(1,keepdims=True)
    print("anchored_bias_fold0",round(evaluate(data,idx0,z0,d0["success"],d0["candidates"][:,:,0],tr0),6))

    # Per-class pair-selector gates test whether the selectors contribute to
    # different fault classes. Start from the globally selected 25% pair mix.
    class_gate = np.full(7, .25)
    def gated(sources_, gate):
        base_rank = (2/3) * sources_[0] + (1/3) * sources_[1]
        q_ = (1-gate[None, :]) * base_rank + gate[None, :] * sources_[2]
        return q_ / np.maximum(q_.sum(1, keepdims=True), 1e-12)
    for _ in range(2):
        for cls in range(7):
            options = []
            for value in (0, .25, .5, .75, 1):
                trial = class_gate.copy(); trial[cls] = value
                q_ = gated(sources, trial)
                options.append((evaluate(data,idx[selection],q_[selection],dev["success"][selection],
                    dev["candidates"][selection,:,0],train), value))
            class_gate[cls] = max(options)[1]
    q_class = gated(sources, class_gate)
    print("class_gates", class_gate.tolist(),
        "selection", round(evaluate(data,idx[selection],q_class[selection],dev["success"][selection],dev["candidates"][selection,:,0],train),6),
        "full", round(evaluate(data,idx,q_class,dev["success"],dev["candidates"][:,:,0],train),6),
        "check", round(evaluate(data,idx[check],q_class[check],dev["success"][check],dev["candidates"][check,:,0],train),6))
    q_class0 = gated([d0["selector_probabilities"],t0["selector_probabilities"],p0f["standard"]],class_gate)
    print("class_gates_fold0",round(evaluate(data,idx0,q_class0,d0["success"],d0["candidates"][:,:,0],tr0),6))
    sources0 = [d0["selector_probabilities"],t0["selector_probabilities"],p0f["standard"]]
    dual_gate = np.full(7,.25)
    for _ in range(2):
        for cls in range(7):
            options=[]
            for value in (0,.25,.5,.75,1):
                trial=dual_gate.copy(); trial[cls]=value
                qd=gated(sources,trial); q0d=gated(sources0,trial)
                objective=.5*evaluate(data,idx[selection],qd[selection],dev["success"][selection],dev["candidates"][selection,:,0],train)+.5*evaluate(data,idx0,q0d,d0["success"],d0["candidates"][:,:,0],tr0)
                options.append((objective,value))
            dual_gate[cls]=max(options)[1]
    qdual=gated(sources,dual_gate); qdual0=gated(sources0,dual_gate)
    print("dual_gates",dual_gate.tolist(),
        "selection",round(evaluate(data,idx[selection],qdual[selection],dev["success"][selection],dev["candidates"][selection,:,0],train),6),
        "check",round(evaluate(data,idx[check],qdual[check],dev["success"][check],dev["candidates"][check,:,0],train),6),
        "fold0",round(evaluate(data,idx0,qdual0,d0["success"],d0["candidates"][:,:,0],tr0),6))


if __name__ == "__main__":
    main()
