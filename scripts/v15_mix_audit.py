"""Audit fixed v15 detector/selector probability mixtures on exported splits."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.metrics import LABELS, binary_f1, composite, fault_turn_hit_at_k, f1_per_class, macro_f1


HANDOFF = ROOT / "handoff/v15/assembled"
EXPORT = HANDOFF / "evaluation_export"
OUTPUTS = HANDOFF / "outputs"


def load_slice(name, prediction_file, npz_file):
    truth = pd.read_csv(EXPORT / f"{name}_truth.csv")
    membership = pd.read_csv(EXPORT / f"{name}_membership.csv")
    saved = np.load(npz_file)
    np.testing.assert_array_equal(truth.run_id, membership.run_id)
    np.testing.assert_array_equal(saved["idx"], membership.official_row_index)
    p_blend = saved["label_probabilities"].astype(np.float64)
    selector = saved["selector_probabilities"].astype(np.float64)
    base = 2 * p_blend - selector
    np.testing.assert_allclose(base.sum(1), 1, atol=1e-6)
    return {"truth": truth, "membership": membership, "base": base,
        "selector": selector, "success": saved["success"],
        "candidates": saved["candidates"]}


def predict(slice_, gates):
    q = (1 - gates[None, :]) * slice_["base"] + gates[None, :] * slice_["selector"]
    q /= np.maximum(q.sum(1, keepdims=True), 1e-12)
    return q


def score_probability(slice_, q):
    labels_i = q.argmax(1)
    labels = np.asarray(LABELS)[labels_i]
    turns = slice_["candidates"][np.arange(len(q)), labels_i]
    truth = slice_["truth"]
    y, success, fault = truth.label.to_numpy(), truth.success.to_numpy(), truth.fault_turn.to_numpy()
    hard = slice_["membership"].robustness_proxy.to_numpy().astype(bool)
    scores = {"macro_f1": macro_f1(y, labels),
        "robustness_f1": macro_f1(y[hard], labels[hard]),
        "success_f1": binary_f1(success, slice_["success"]),
        "fault_turn_hit2": fault_turn_hit_at_k(y, labels, fault, turns)}
    scores["score_proxy"] = composite(scores)
    return scores


def evaluate(slice_, gates):
    return score_probability(slice_, predict(slice_, gates))


def apply_bias(q, bias):
    z = np.log(np.maximum(q, 1e-12)) + bias[None, :]
    z -= z.max(1, keepdims=True)
    result = np.exp(z)
    return result / result.sum(1, keepdims=True)


def main():
    dev_path = OUTPUTS / "pair_selector_v15/validation.npz"
    ctl_path = OUTPUTS / "pair_selector_v15/fold0/validation.npz"
    dev = load_slice("development_fold1", None, dev_path)
    ctl = load_slice("fold0", None, ctl_path)
    # Map the check-half rows into the original development prediction order.
    dev_ids = dev["membership"].official_row_index.to_numpy()
    check_truth = pd.read_csv(EXPORT / "check_half_truth.csv")
    check_membership = pd.read_csv(EXPORT / "check_half_membership.csv")
    np.testing.assert_array_equal(check_truth.run_id, check_membership.run_id)
    check_ids = check_membership.official_row_index.to_numpy()
    position = {int(value): i for i, value in enumerate(dev_ids)}
    check_pos = np.asarray([position[int(value)] for value in check_ids])
    check = {"truth": check_truth, "membership": check_membership,
        "base": dev["base"][check_pos], "selector": dev["selector"][check_pos],
        "success": dev["success"][check_pos], "candidates": dev["candidates"][check_pos]}
    check_ids_set = set(check_ids.tolist())
    sel_pos = np.asarray([i for i, value in enumerate(dev_ids) if int(value) not in check_ids_set])
    selection = {key: (value.iloc[sel_pos].reset_index(drop=True) if key in ("truth", "membership")
        else value[sel_pos]) for key, value in dev.items()}

    trials = []
    for weight in (0, .25, .5, .75, 1):
        gates = np.full(7, weight)
        trials.append((evaluate(selection, gates)["score_proxy"], weight))
    chosen_weight = max(trials)[1]
    global_gates = np.full(7, chosen_weight)
    print("global_trials", [(w, round(s, 6)) for s, w in trials])
    print("global_weight", chosen_weight,
        "selection", round(evaluate(selection, global_gates)["score_proxy"], 6),
        "development", round(evaluate(dev, global_gates)["score_proxy"], 6),
        "check", round(evaluate(check, global_gates)["score_proxy"], 6),
        "fold0", round(evaluate(ctl, global_gates)["score_proxy"], 6))

    gates = global_gates.copy()
    for _ in range(2):
        for cls in range(7):
            candidates = []
            for value in (0, .25, .5, .75, 1):
                trial = gates.copy()
                trial[cls] = value
                candidates.append((evaluate(selection, trial)["score_proxy"], value))
            gates[cls] = max(candidates)[1]
    print("class_gates", gates.tolist(),
        "selection", round(evaluate(selection, gates)["score_proxy"], 6),
        "development", round(evaluate(dev, gates)["score_proxy"], 6),
        "check", round(evaluate(check, gates)["score_proxy"], 6),
        "fold0", round(evaluate(ctl, gates)["score_proxy"], 6))
    effective_gates = .125 + .75*gates
    print("effective_vector_gates", effective_gates.tolist(),
        "selection",round(evaluate(selection,effective_gates)["score_proxy"],6),
        "development",round(evaluate(dev,effective_gates)["score_proxy"],6),
        "check",round(evaluate(check,effective_gates)["score_proxy"],6),
        "fold0",round(evaluate(ctl,effective_gates)["score_proxy"],6))
    p_global = predict(dev, global_gates)
    p_class = predict(dev, gates)
    p0_global = predict(ctl, global_gates)
    p0_class = predict(ctl, gates)
    for alpha in (.25, .5, .75):
        p_dev = (1-alpha)*p_global+alpha*p_class
        p0 = (1-alpha)*p0_global+alpha*p0_class
        print("soft_gate", alpha,
            "selection", round(score_probability(selection,p_dev[sel_pos])["score_proxy"],6),
            "development", round(score_probability(dev,p_dev)["score_proxy"],6),
            "check", round(score_probability(check,p_dev[check_pos])["score_proxy"],6),
            "fold0", round(score_probability(ctl,p0)["score_proxy"],6))
        if alpha == .75:
            for name, result in (("dev",score_probability(dev,p_dev)),
                    ("check",score_probability(check,p_dev[check_pos])),
                    ("fold0",score_probability(ctl,p0))):
                print("soft_gate_components",name,{k:round(v,6) for k,v in result.items()})

    dual = global_gates.copy()
    for _ in range(2):
        for cls in range(7):
            trials = []
            for value in (0,.25,.5,.75,1):
                candidate = dual.copy(); candidate[cls] = value
                objective = .5*evaluate(selection,candidate)["score_proxy"] + .5*evaluate(ctl,candidate)["score_proxy"]
                trials.append((objective,value))
            dual[cls] = max(trials)[1]
    print("selection_control_gates",dual.tolist(),
        "selection",round(evaluate(selection,dual)["score_proxy"],6),
        "development",round(evaluate(dev,dual)["score_proxy"],6),
        "check",round(evaluate(check,dual)["score_proxy"],6),
        "fold0",round(evaluate(ctl,dual)["score_proxy"],6))

    bias = np.zeros(7)
    for _ in range(2):
        for cls in range(7):
            candidates = []
            for value in np.arange(-.3,.301,.05):
                trial = bias.copy(); trial[cls] = value
                q = apply_bias(predict(selection,global_gates),trial)
                candidates.append((score_probability(selection,q)["score_proxy"],value))
            bias[cls] = max(candidates)[1]
    qdev = apply_bias(predict(dev,global_gates),bias)
    qctl = apply_bias(predict(ctl,global_gates),bias)
    qchk = qdev[check_pos]
    print("logit_bias",np.round(bias,2).tolist(),
        "selection",round(score_probability(selection,apply_bias(predict(selection,global_gates),bias))["score_proxy"],6),
        "development",round(score_probability(dev,qdev)["score_proxy"],6),
        "check",round(score_probability(check,qchk)["score_proxy"],6),
        "fold0",round(score_probability(ctl,qctl)["score_proxy"],6))

    ranker_dev = np.load(OUTPUTS/"stack_v9/validation.npz")
    ranker_ctl = np.load(OUTPUTS/"stack_v9/fold0/validation.npz")
    np.testing.assert_array_equal(ranker_dev["idx"],dev["membership"].official_row_index)
    np.testing.assert_array_equal(ranker_ctl["idx"],ctl["membership"].official_row_index)
    qdev_gate = .25*p_global+.75*p_class
    qctl_gate = .25*p0_global+.75*p0_class
    # Use the archived v9 final probability distributions. The raw fold-0
    # ranker output was not retained, while the complete v9 output was.
    rank_dev = ranker_dev["label_probabilities"]
    rank_ctl = ranker_ctl["label_probabilities"]
    rank_trials=[]
    for weight in (0,.25,.5,.75,1):
        q=(1-weight)*qdev_gate+weight*rank_dev
        rank_trials.append((score_probability(selection,q[sel_pos])["score_proxy"],weight))
    rank_weight=max(rank_trials)[1]
    qdev_rank=(1-rank_weight)*qdev_gate+rank_weight*rank_dev
    qctl_rank=(1-rank_weight)*qctl_gate+rank_weight*rank_ctl
    print("v9_ranker_blend_trials",[(w,round(s,6)) for s,w in rank_trials])
    print("v9_ranker_blend",rank_weight,
        "development",round(score_probability(dev,qdev_rank)["score_proxy"],6),
        "check",round(score_probability(check,qdev_rank[check_pos])["score_proxy"],6),
        "fold0",round(score_probability(ctl,qctl_rank)["score_proxy"],6))
    rank_gates=np.full(7,rank_weight)
    for _ in range(2):
        for cls in range(7):
            choices=[]
            for value in (0,.25,.5,.75,1):
                trial=rank_gates.copy(); trial[cls]=value
                q=(1-trial[None,:])*qdev_gate+trial[None,:]*rank_dev
                q/=np.maximum(q.sum(1,keepdims=True),1e-12)
                choices.append((score_probability(selection,q[sel_pos])["score_proxy"],value))
            rank_gates[cls]=max(choices)[1]
    qdev_class=(1-rank_gates[None,:])*qdev_gate+rank_gates[None,:]*rank_dev
    qdev_class/=np.maximum(qdev_class.sum(1,keepdims=True),1e-12)
    qctl_class=(1-rank_gates[None,:])*qctl_gate+rank_gates[None,:]*rank_ctl
    qctl_class/=np.maximum(qctl_class.sum(1,keepdims=True),1e-12)
    print("v9_class_gates",rank_gates.tolist(),
        "selection",round(score_probability(selection,qdev_class[sel_pos])["score_proxy"],6),
        "development",round(score_probability(dev,qdev_class)["score_proxy"],6),
        "check",round(score_probability(check,qdev_class[check_pos])["score_proxy"],6),
        "fold0",round(score_probability(ctl,qctl_class)["score_proxy"],6))
    for name,result in (("development",score_probability(dev,qdev_class)),
            ("check",score_probability(check,qdev_class[check_pos])),
            ("fold0",score_probability(ctl,qctl_class))):
        print("v9_class_components",name,{k:round(v,6) for k,v in result.items()})
    for name,slice_,q in (("check",check,qdev_class[check_pos]),("fold0",ctl,qctl_class)):
        labels=np.asarray(LABELS)[q.argmax(1)]
        print("v9_class_f1",name,{k:round(v,4) for k,v in f1_per_class(
            slice_["truth"].label.to_numpy(),labels,LABELS).items()})

    # Save a separate, reviewable prediction candidate; source checkpoints
    # remain untouched.
    folder=ROOT/"outputs/v15-class-gate-candidate"
    folder.mkdir(parents=True,exist_ok=True)
    def export(slice_,q,name):
        label_i=q.argmax(1)
        table=pd.DataFrame({"run_id":slice_["truth"].run_id,
            "label":np.asarray(LABELS)[label_i],
            "success":slice_["success"],
            "fault_turn":slice_["candidates"][np.arange(len(q)),label_i]})
        table.to_csv(folder/f"{name}.csv",index=False)
        return score_probability(slice_,q)
    candidate_scores={
        "selection_half":export(selection,qdev_class[sel_pos],"selection_half"),
        "development_fold1":export(dev,qdev_class,"development_fold1"),
        "check_half":export(check,qdev_class[check_pos],"check_half"),
        "fold0":export(ctl,qctl_class,"fold0")}
    config={"source":"unchanged v15 + archived stack_v9 output distributions",
        "selection_seed":12,"selection_objective":"score_proxy",
        "v15_pair_gate":{"initial_global_pair_weight":.5,"class_gates":[1,.5,1,0,.5,.5,.5],"gate_strength":.75},
        "v9_output_class_gates":rank_gates.tolist(),
        "labels_order":LABELS,
        "metrics":candidate_scores}
    (folder/"configuration_and_metrics.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    print("candidate_saved",str(folder))

    # Paired row bootstrap on check-half; descriptive only, not a new split.
    q_base = predict(dev,global_gates)[check_pos]
    q_cand = qdev_class[check_pos]
    truth = check["truth"]
    y = truth.label.to_numpy()
    success = check["success"]
    fault = truth.fault_turn.to_numpy()
    hard = check["membership"].robustness_proxy.to_numpy().astype(bool)
    cand_turns = check["candidates"]
    lab_base_i, lab_cand_i = q_base.argmax(1), q_cand.argmax(1)
    pred_base, pred_cand = np.asarray(LABELS)[lab_base_i], np.asarray(LABELS)[lab_cand_i]
    turn_base = cand_turns[np.arange(len(y)),lab_base_i]
    turn_cand = cand_turns[np.arange(len(y)),lab_cand_i]
    rng = np.random.default_rng(20261015)
    deltas=[]
    for _ in range(2000):
        sample = rng.integers(0,len(y),len(y))
        def sampled_score(pred,turn):
            mask=hard[sample]
            scores={"macro_f1":macro_f1(y[sample],pred[sample]),
                "robustness_f1":macro_f1(y[sample][mask],pred[sample][mask]),
                "success_f1":binary_f1(truth.success.to_numpy()[sample],success[sample]),
                "fault_turn_hit2":fault_turn_hit_at_k(y[sample],pred[sample],fault[sample],turn[sample])}
            return composite(scores)
        deltas.append(sampled_score(pred_cand,turn_cand)-sampled_score(pred_base,turn_base))
    print("check_paired_bootstrap_delta",{"median":round(float(np.median(deltas)),6),
        "ci95":[round(float(v),6) for v in np.quantile(deltas,[.025,.975])],
        "share_positive":round(float(np.mean(np.asarray(deltas)>0)),4)})


if __name__ == "__main__":
    main()
