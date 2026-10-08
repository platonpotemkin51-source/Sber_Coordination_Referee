"""Small probability-only search, selected on development selection-half only.

No classifier is fitted to the check-half or fold zero. Fold zero is evaluated
once with the frozen choice, and its control tree was trained on 18k rows while
the ranker/neural predictions come from models trained on 16k rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from referee.data import load_data, fixed_folds, concatenate, LABELS
from scripts.experiment import report
from evaluation.metrics import binary_f1


def temperature(p, t):
    logp = np.log(np.maximum(p, 1e-12)) / t
    p = np.exp(logp-logp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def fitted_temperature(p, y, select):
    def loss(t):
        pp = temperature(p[select], t)
        return float(-np.log(np.maximum(pp[np.arange(len(select)), y[select]], 1e-12)).mean())
    fit = minimize_scalar(loss, bounds=(.4, 3.), method='bounded')
    return float(fit.x)


def apply_parameters(predictions, params):
    p = {name: temperature(values, params['temperatures'][name])
         for name, values in predictions.items()}
    neural = p['neural']
    if params['method'] == 'threeway':
        return sum(weight*p[name] for name, weight in params['weights'].items())
    base = p[params['source']]
    weight = params['neural_weight']
    if params['method'].startswith('confidence'):
        if params['method'] == 'confidence_margin':
            b = np.sort(base, axis=1); n = np.sort(neural, axis=1)
            better = (n[:,-1]-n[:,-2]) > (b[:,-1]-b[:,-2])
        else:
            better = -(neural*np.log(np.maximum(neural, 1e-12))).sum(1) < -(base*np.log(np.maximum(base, 1e-12))).sum(1)
        weight = np.where(better, weight, 1-weight)[:,None]
    if params['method'] == 'geometric':
        logits = (1-weight)*np.log(np.maximum(base, 1e-12))+weight*np.log(np.maximum(neural, 1e-12))
        merged = np.exp(logits-logits.max(1, keepdims=True))
        return merged/merged.sum(1, keepdims=True)
    return (1-weight)*base+weight*neural


def bundle(folder):
    folders = {'tree': folder['tree'], 'ranked': folder['ranked'], 'neural': folder['neural']}
    archives = {name: np.load(ROOT/'outputs'/path/'validation.npz') for name, path in folders.items()}
    idx = archives['tree']['idx']
    for other in archives.values():
        np.testing.assert_array_equal(idx, other['idx'])
    candidates = np.load(ROOT/'outputs'/folder['tree']/'turns.npz')['candidates']
    return idx, {name: archive['label_probabilities'] for name, archive in archives.items()}, archives['neural']['success_probabilities'], candidates


def diagnostics(y, predictions, candidates, fault):
    argmax = {name: p.argmax(1) for name, p in predictions.items()}
    neural = argmax['neural']; tree = argmax['tree']; ranked = argmax['ranked']
    out = {'rows': int(len(y)), 'all_three_wrong': int(((neural!=y)&(tree!=y)&(ranked!=y)).sum()),
           'oracle_any_class_accuracy': float(((neural==y)|(tree==y)|(ranked==y)).mean())}
    masks = {'tree_neural_disagree': tree!=neural, 'tree_neural_agree': tree==neural,
             'ranked_neural_disagree': ranked!=neural}
    for title, mask in masks.items():
        out[title] = {'n': int(mask.sum()), **{name: float((labels[mask]==y[mask]).mean()) if mask.any() else 0. for name, labels in argmax.items()}}
    out['per_class_errors'] = {name: {label: int(((y==class_id)&(pred.argmax(1)!=y)).sum()) for class_id, label in enumerate(LABELS)} for name, pred in predictions.items()}
    nonclean = y>0
    out['tree_onset_hit_with_oracle_class'] = float((np.abs(candidates[np.arange(len(y)), y][nonclean]-fault[nonclean])<=2).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='outputs/advanced_blend_v11')
    args = ap.parse_args()
    data = load_data('data/competition/train.csv'); folds = fixed_folds(data)
    extra = load_data('data/train.csv')
    excluded = set(data['groups'][folds<2])
    train = np.concatenate([np.flatnonzero(folds>1), np.array([i for i,g in enumerate(extra['groups']) if g not in excluded])+len(folds)])
    train0 = np.concatenate([np.flatnonzero(folds!=0), np.array([i for i,g in enumerate(extra['groups']) if g not in set(data['groups'][folds==0])])+len(folds)])
    data = concatenate(data, extra)
    idx, predictions, ps, candidates = bundle({'tree':'relations_v5','ranked':'stack_v9','neural':'sequence_v6'})
    assert (folds[idx]==1).all()
    select, check = train_test_split(np.arange(len(idx)), test_size=.5, stratify=data['y'][idx], random_state=12)
    y = data['y'][idx]
    raw = {name: 1. for name in predictions}
    calibrated = {name: fitted_temperature(p, y, select) for name,p in predictions.items()}
    threshold = max((.35,.4,.45,.5,.55), key=lambda t: binary_f1(data['success'][idx][select], ps[select]>=t))
    successes = (ps>=threshold).astype(int)
    choices = []
    for temperatures in (raw, calibrated):
        for source in ('tree', 'ranked'):
            for method in ('arithmetic', 'geometric'):
                for weight in (.1,.25,.5,.75,.9):
                    choices.append({'source':source,'method':method,'neural_weight':weight,'temperatures':temperatures})
            for method in ('confidence_margin', 'confidence_entropy'):
                for weight in (.75,.9):
                    choices.append({'source':source,'method':method,'neural_weight':weight,'temperatures':temperatures})
        for weights in ({'tree':.25,'ranked':.25,'neural':.5},{'tree':.1,'ranked':.4,'neural':.5},{'tree':.1,'ranked':.15,'neural':.75}):
            choices.append({'method':'threeway','weights':weights,'temperatures':temperatures})
    records = []
    for parameters in choices:
        p = apply_parameters(predictions, parameters); labels = p.argmax(1)
        turns = candidates[np.arange(len(idx)),labels]
        selected = report(data,idx[select],labels[select],successes[select],turns[select],train)
        records.append({'parameters':parameters,'selection_score_proxy':selected['score_proxy']})
    best = max(records,key=lambda r:r['selection_score_proxy'])['parameters']
    best['success_threshold'] = threshold; best['success_source']='neural'; best['onset_source']='tree'
    p = apply_parameters(predictions,best); labels=p.argmax(1); turns=candidates[np.arange(len(idx)),labels]
    scores = {name: report(data,idx[rows],labels[rows],successes[rows],turns[rows],train) for name,rows in (('selection',select),('check',check),('full',np.arange(len(idx))))}
    # No parameters are selected using these fold zero predictions.
    idx0, pred0, ps0, candidates0 = bundle({'tree':'control_relations_v5','ranked':'stack_v9/fold0','neural':'sequence_v6/fold0'})
    assert (folds[idx0]==0).all()
    p0=apply_parameters(pred0,best); labels0=p0.argmax(1); success0=(ps0>=threshold).astype(int); turns0=candidates0[np.arange(len(idx0)),labels0]
    scores['fold0']=report(data,idx0,labels0,success0,turns0,train0)
    scores['parameters']=best
    scores['candidate_count']=len(records)
    scores['notes']=['Temperatures, blending recipe, and success threshold selected only on 1000-row development selection-half.', 'Check-half and fold zero were evaluated after freezing selection. Fold zero was already opened during earlier research and is not a fresh independent final holdout.', 'Robustness uses a length/topology proxy, not the unavailable official shifted subset.', 'Fold zero control tree trained on 18k rows; ranker and neural models trained on 16k. Development models all use 16k.']
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(scores,indent=2),encoding='utf8')
    (out/'parameters.json').write_text(json.dumps(best,indent=2),encoding='utf8')
    (out/'selection_candidates.json').write_text(json.dumps(sorted(records,key=lambda r:r['selection_score_proxy'],reverse=True),indent=2),encoding='utf8')
    (out/'diagnostics.json').write_text(json.dumps({'development':diagnostics(y,predictions,candidates,data['fault'][idx]),'fold0':diagnostics(data['y'][idx0],pred0,candidates0,data['fault'][idx0])},indent=2),encoding='utf8')
    np.savez_compressed(out/'validation.npz',idx=idx,label_probabilities=p,success_probabilities=ps,labels=labels,success=successes,turns=turns,folds=folds)
    np.savez_compressed(out/'fold0.npz',idx=idx0,label_probabilities=p0,success_probabilities=ps0,labels=labels0,success=success0,turns=turns0,folds=folds)
    print(json.dumps(scores,indent=2),flush=True)


if __name__=='__main__':
    main()
