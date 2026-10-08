"""Refine the frozen sequence model with low learning rate and weight averaging.

Training rows and normalization remain exactly those of sequence_v6.
Epoch selection uses only selection-half of development fold 1.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from referee.data import load_data, fixed_folds, concatenate, CODE_HASH
from referee.relations import relation_features, RELATIONS_HASH
from referee.sequence import SequenceNetwork, signed_log, batch_arrays, predict
from scripts.experiment import report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='sequence_refine_v12')
    ap.add_argument('--init', default='outputs/sequence_v6/model.pt')
    ap.add_argument('--train', default='data/competition/train.csv')
    ap.add_argument('--extra-train', default='data/train.csv')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--learning-rate', type=float, default=6e-5)
    ap.add_argument('--ema-decay', type=float, default=.995)
    ap.add_argument('--label-smoothing', type=float, default=0.)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = ap.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(43)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    rng = np.random.default_rng(43)
    checkpoint = torch.load(args.init, map_location='cpu', weights_only=False)
    if checkpoint['features_code_hash'] != CODE_HASH or checkpoint['relations_hash'] != RELATIONS_HASH:
        raise ValueError('Initial checkpoint feature provenance mismatch')
    data = relation_features(load_data(args.train), args.train)
    folds = fixed_folds(data)
    valid = np.flatnonzero(folds == 1)
    training = np.flatnonzero(folds > 1)
    excluded = set(data['groups'][folds < 2])
    if args.extra_train:
        extra = relation_features(load_data(args.extra_train), args.extra_train)
        selected = np.array([i for i, g in enumerate(extra['groups']) if g not in excluded]) + len(data['y'])
        training = np.concatenate([training, selected])
        data = concatenate(data, extra)
    assert not (set(data['groups'][training]) & set(data['groups'][valid]))
    select, check = train_test_split(valid, test_size=.5, stratify=data['y'][valid], random_state=12)
    stats = checkpoint['normalization']
    run = np.clip((signed_log(data['X'][checkpoint['run_columns']].to_numpy(dtype=np.float32))-stats['run_mean'])/stats['run_std'], -8, 8)
    columns = [data['turn_columns'].index(c) for c in checkpoint['turn_columns']]
    turns = np.clip((signed_log(data['turn_X'][:, columns])-stats['turn_mean'])/stats['turn_std'], -8, 8)
    network = SequenceNetwork(run.shape[1], turns.shape[1], checkpoint['width']).to(device)
    network.load_state_dict(checkpoint['state'])
    averaged = copy.deepcopy(network).eval()
    for parameter in averaged.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate, weight_decay=.02)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.learning_rate*.1)
    folder = ROOT/'outputs'/args.name
    folder.mkdir(parents=True, exist_ok=True)
    best = -1
    history = []
    # Include the unmodified checkpoint, so refinement never selects a worse
    # epoch on the selection subset merely because all updates are harmful.
    for epoch in range(args.epochs+1):
        epoch_started = time.monotonic()
        losses = []
        if epoch:
            network.train()
            order = rng.permutation(training)
            for start in range(0, len(order), args.batch_size):
                selected = order[start:start+args.batch_size]
                batch = tuple(x.to(device) for x in batch_arrays(run, turns, data['offsets'], selected))
                yn = data['y'][selected].astype(np.int64)
                y = torch.from_numpy(yn).to(device)
                success = torch.from_numpy(data['success'][selected].astype(np.float32)).to(device)
                labels, ps, onset = network(*batch)
                loss = torch.nn.functional.cross_entropy(labels, y, label_smoothing=args.label_smoothing) + .5*torch.nn.functional.binary_cross_entropy_with_logits(ps, success)
                faulty = y > 0
                if faulty.any():
                    local = onset[faulty, :, y[faulty]-1]
                    target = torch.from_numpy(data['fault'][selected][yn > 0].astype(np.int64)).to(device)
                    loss = loss + .3*torch.nn.functional.cross_entropy(local, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 2)
                optimizer.step()
                with torch.no_grad():
                    for average, current in zip(averaged.parameters(), network.parameters()):
                        average.lerp_(current, 1-args.ema_decay)
                losses.append(float(loss.detach()))
            schedule.step()
        p, ps, candidates = predict(averaged, run, turns, data['offsets'], select)
        labels = p.argmax(1)
        metrics = report(data, select, labels, (ps >= .45).astype(int), candidates[np.arange(len(select)), labels], training)
        log = {'epoch': epoch, 'loss': float(np.mean(losses)) if losses else None, 'seconds': round(time.monotonic()-epoch_started, 2), 'learning_rate': optimizer.param_groups[0]['lr'], 'selection_score_proxy': metrics['score_proxy']}
        history.append(log)
        print(json.dumps(log), flush=True)
        if metrics['score_proxy'] > best:
            best = metrics['score_proxy']
            result = dict(checkpoint)
            result.update(state={k:v.detach().cpu() for k,v in averaged.state_dict().items()}, epoch=epoch, initialization_sha256=hashlib.sha256(Path(args.init).read_bytes()).hexdigest(), refinement=vars(args), refinement_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            torch.save(result, folder/'model.pt')
    selected = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
    averaged.load_state_dict(selected['state'])
    p, ps, candidates = predict(averaged, run, turns, data['offsets'], valid)
    labels = p.argmax(1)
    success = (ps >= .45).astype(int)
    onset = candidates[np.arange(len(valid)), labels]
    metrics = report(data, valid, labels, success, onset, training)
    positions = np.flatnonzero(np.isin(valid, check))
    metrics['check'] = report(data, valid[positions], labels[positions], success[positions], onset[positions], training)
    metrics.update(configuration=vars(args), selected_epoch=selected['epoch'], history=history, elapsed_seconds=time.monotonic()-started, features_code_hash=CODE_HASH, relations_hash=RELATIONS_HASH, training_rows=len(training))
    (folder/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    np.savez_compressed(folder/'validation.npz', idx=valid, label_probabilities=p, success_probabilities=ps, labels=labels, success=success, turns=onset, candidates=candidates, folds=folds)
    print(json.dumps({k:v for k,v in metrics.items() if k!='history'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
