"""Train a fresh multitask transformer with EMA on fixed, disjoint run splits.

Epoch selection uses only half of development fold 1. Fold 0 is evaluated
after selecting the checkpoint; it was already opened in earlier experiments.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F

from referee.data import CODE_HASH, concatenate, fixed_folds, load_data
from referee.relations import RELATIONS_HASH, relation_features
from referee.sequence import SequenceNetwork, batch_arrays, normalize_training, predict
from scripts.experiment import report


def make_network(run_features, turn_features, *, width=128, layers=3, dropout=.15):
    """Preserve SequenceNetwork head/state names while configuring its encoder."""
    if width <= 0 or width % 4 or layers < 1 or not 0 <= dropout < 1:
        raise ValueError('width must be a positive multiple of 4, layers >= 1, and dropout in [0, 1)')
    network = SequenceNetwork(run_features, turn_features, width)
    layer = nn.TransformerEncoderLayer(
        width, 4, dim_feedforward=width * 3, dropout=dropout,
        batch_first=True, norm_first=True, activation='gelu')
    network.sequence = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
    for module in network.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout
    return network


def prepare_splits(primary, extra=None):
    """Append eligible demo runs without admitting reserved content groups."""
    folds = fixed_folds(primary)
    training = np.flatnonzero(folds > 1)
    development = np.flatnonzero(folds == 1)
    control = np.flatnonzero(folds == 0)
    data = primary
    excluded = set(primary['groups'][folds < 2])
    demo_training_rows = 0
    if extra is not None:
        eligible = np.array([i for i, group in enumerate(extra['groups'])
                             if group not in excluded], dtype=np.int64)
        demo_training_rows = len(eligible)
        training = np.concatenate([training, eligible + len(primary['y'])])
        data = concatenate(primary, extra)
    selection, check = train_test_split(
        development, test_size=.5, stratify=data['y'][development], random_state=12)
    training_groups = set(data['groups'][training])
    if training_groups & excluded:
        raise ValueError('Training contains a reserved fold 0/1 content group')
    if set(selection) & set(check) or set(selection) | set(check) != set(development):
        raise ValueError('Development selection/check split is not a disjoint partition')
    return data, {
        'training': training, 'development': development, 'selection': selection,
        'check': check, 'fold0': control, 'folds': folds,
        'official_training_rows': int((folds > 1).sum()),
        'demo_training_rows': int(demo_training_rows)}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def provenance(data, splits, sources, architecture, configuration):
    run_columns = list(data['X'].columns)
    turn_columns = list(data['turn_columns'])
    groups = data['groups']
    partition = {}
    for name in ('training', 'development', 'selection', 'check', 'fold0'):
        indexes = splits[name]
        partition[name] = {
            'rows': int(len(indexes)), 'indices_sha256': digest_json(indexes.tolist()),
            'groups_sha256': digest_json(sorted(groups[indexes].tolist()))}
    return {
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sequence_module_sha256': hashlib.sha256((ROOT / 'referee/sequence.py').read_bytes()).hexdigest(),
        'features_code_hash': CODE_HASH, 'relations_hash': RELATIONS_HASH,
        'schema_sha256': digest_json({'run_columns': run_columns, 'turn_columns': turn_columns}),
        'sources': sources, 'partitions': partition,
        'official_training_rows': splits['official_training_rows'],
        'demo_training_rows': splits['demo_training_rows'],
        'reserved_training_group_overlap': 0,
        'architecture': architecture, 'configuration': configuration,
        'initialization': 'fresh_random_weights', 'random_seed': configuration['seed'],
        'fold_seed': 20261007, 'selection_seed': 12,
        'normalization_fit_partition': 'training', 'checkpoint_selection_partition': 'selection',
        'checkpoint_weights': 'EMA after each optimizer step',
        'fold0_note': 'Excluded from fitting and epoch selection; already opened during earlier model development.'}


@torch.no_grad()
def update_ema(average, network, decay):
    for averaged, current in zip(average.parameters(), network.parameters(), strict=True):
        averaged.lerp_(current.detach(), 1 - decay)
    for averaged, current in zip(average.buffers(), network.buffers(), strict=True):
        if averaged.is_floating_point():
            averaged.lerp_(current.detach(), 1 - decay)
        else:
            averaged.copy_(current.detach())


def multitask_loss(network, batch, data, indexes, device, label_smoothing):
    selected_y = data['y'][indexes].astype(np.int64)
    y = torch.from_numpy(selected_y).to(device)
    success = torch.from_numpy(data['success'][indexes].astype(np.float32)).to(device)
    labels, outcome, onset = network(*batch)
    loss = F.cross_entropy(labels, y, label_smoothing=label_smoothing)
    loss = loss + .5 * F.binary_cross_entropy_with_logits(outcome, success)
    faulty = y > 0
    if faulty.any():
        local = onset[faulty].gather(
            2, (y[faulty] - 1)[:, None, None].expand(-1, onset.shape[1], 1)).squeeze(2)
        target = torch.from_numpy(data['fault'][indexes][selected_y > 0].astype(np.int64)).to(device)
        loss = loss + .3 * F.cross_entropy(local, target)
    return loss


def evaluate_predictions(data, indexes, probabilities, outcomes, candidates, training, threshold):
    labels = probabilities.argmax(1)
    success = (outcomes >= threshold).astype(int)
    turns = candidates[np.arange(len(indexes)), labels]
    return report(data, indexes, labels, success, turns, training), labels, success, turns


def save_evaluation(folder, indexes, probabilities, outcomes, candidates, folds, metrics, predictions):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    labels, success, turns = predictions
    np.savez_compressed(
        folder / 'validation.npz', idx=indexes, label_probabilities=probabilities,
        success_probabilities=outcomes, labels=labels, success=success, turns=turns,
        candidates=candidates, folds=folds)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='fresh_sequence_v15')
    parser.add_argument('--train', default='data/competition/train.csv')
    parser.add_argument('--extra-train', default='data/train.csv')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=.15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=6e-4)
    parser.add_argument('--weight-decay', type=float, default=.03)
    parser.add_argument('--ema-decay', type=float, default=.99)
    parser.add_argument('--label-smoothing', type=float, default=.02)
    parser.add_argument('--success-threshold', type=float, default=.45)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error('epochs/batch-size/learning-rate must be positive and weight-decay nonnegative')
    if not 0 <= args.ema_decay < 1 or not 0 <= args.label_smoothing < 1:
        parser.error('ema-decay and label-smoothing must lie in [0, 1)')
    if not 0 <= args.success_threshold <= 1:
        parser.error('success-threshold must lie in [0, 1]')
    return args


def main():
    args = parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable in this Python environment')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    primary = relation_features(load_data(args.train), args.train)
    sources = [{'path': str(args.train), 'sha256': primary['source_sha256'], 'rows': len(primary['y'])}]
    extra = None
    if args.extra_train:
        extra = relation_features(load_data(args.extra_train), args.extra_train)
        sources.append({'path': str(args.extra_train), 'sha256': extra['source_sha256'], 'rows': len(extra['y'])})
    data, splits = prepare_splits(primary, extra)
    del primary, extra
    training = splits['training']
    development = splits['development']
    selection = splits['selection']
    lengths = np.diff(data['offsets'])
    faulty = data['y'] > 0
    if np.any(data['fault'][faulty] < 0) or np.any(data['fault'][faulty] >= lengths[faulty]):
        raise ValueError('Faulty training/validation run has an out-of-bounds onset label')
    run, turns, normalization = normalize_training(data, training)
    architecture = {'width': args.width, 'layers': args.layers, 'dropout': args.dropout}
    network = make_network(run.shape[1], turns.shape[1], **architecture).to(device)
    average = copy.deepcopy(network).eval()
    average.requires_grad_(False)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=.1 * args.learning_rate)
    folder = ROOT / 'outputs' / args.name
    folder.mkdir(parents=True, exist_ok=True)
    certificate = provenance(data, splits, sources, architecture, vars(args))
    (folder / 'provenance.json').write_text(json.dumps(certificate, indent=2), encoding='utf-8')
    print(json.dumps({'train_rows': len(training), 'selection_rows': len(selection),
                      'check_rows': len(splits['check']), 'fold0_rows': len(splits['fold0']),
                      'architecture': architecture, 'device': args.device,
                      'initialization': 'fresh_random_weights'}), flush=True)
    best_score = -math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        learning_rate = optimizer.param_groups[0]['lr']
        network.train()
        order = rng.permutation(training)
        total_loss = 0.
        seen = 0
        for start_batch in range(0, len(order), args.batch_size):
            indexes = order[start_batch:start_batch + args.batch_size]
            batch = tuple(tensor.to(device) for tensor in batch_arrays(run, turns, data['offsets'], indexes))
            loss = multitask_loss(network, batch, data, indexes, device, args.label_smoothing)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 2)
            optimizer.step()
            update_ema(average, network, args.ema_decay)
            total_loss += float(loss.detach()) * len(indexes)
            seen += len(indexes)
        probabilities, outcomes, candidates = predict(
            average, run, turns, data['offsets'], selection, batch_size=args.batch_size)
        selected_metrics, *_ = evaluate_predictions(
            data, selection, probabilities, outcomes, candidates, training, args.success_threshold)
        score = selected_metrics['score_proxy']
        if score > best_score:
            best_score = score
            torch.save({
                'state': {name: value.detach().cpu().clone() for name, value in average.state_dict().items()},
                'normalization': normalization, 'run_columns': list(data['X'].columns),
                'turn_columns': list(data['turn_columns']), 'width': args.width,
                'architecture': architecture, 'epoch': epoch, 'seed': args.seed,
                'features_code_hash': CODE_HASH, 'relations_hash': RELATIONS_HASH,
                'provenance': certificate, 'selection_score_proxy': score,
                'ema_decay': args.ema_decay, 'success_threshold': args.success_threshold}, folder / 'model.pt')
        scheduler.step()
        entry = {'epoch': epoch, 'loss': total_loss / seen, 'learning_rate': learning_rate,
                 'seconds': round(time.monotonic() - epoch_started, 2),
                 'selection_score_proxy': score, 'best_selection_score_proxy': best_score}
        history.append(entry)
        (folder / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(entry), flush=True)

    # The best epoch and architecture are frozen before check/full/fold 0 scores.
    checkpoint = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
    average.load_state_dict(checkpoint['state'])
    probabilities, outcomes, candidates = predict(
        average, run, turns, data['offsets'], development, batch_size=args.batch_size)
    metrics, *predictions = evaluate_predictions(
        data, development, probabilities, outcomes, candidates, training, args.success_threshold)
    for name in ('selection', 'check'):
        positions = np.flatnonzero(np.isin(development, splits[name]))
        indexes = development[positions]
        subset_metrics, *subset_predictions = evaluate_predictions(
            data, indexes, probabilities[positions], outcomes[positions], candidates[positions],
            training, args.success_threshold)
        metrics[name] = subset_metrics
        save_evaluation(folder / name, indexes, probabilities[positions], outcomes[positions],
                        candidates[positions], splits['folds'], subset_metrics, subset_predictions)
    control = splits['fold0']
    p0, ps0, candidates0 = predict(average, run, turns, data['offsets'], control, batch_size=args.batch_size)
    control_metrics, *control_predictions = evaluate_predictions(
        data, control, p0, ps0, candidates0, training, args.success_threshold)
    control_metrics.update(selected_epoch=checkpoint['epoch'], note=certificate['fold0_note'])
    save_evaluation(folder / 'fold0', control, p0, ps0, candidates0,
                    splits['folds'], control_metrics, control_predictions)
    metrics.update(
        configuration=vars(args), elapsed_seconds=time.monotonic() - started,
        selected_epoch=checkpoint['epoch'], history=history, provenance=certificate,
        features_code_hash=CODE_HASH, relations_hash=RELATIONS_HASH, fold0=control_metrics)
    save_evaluation(folder, development, probabilities, outcomes, candidates,
                    splits['folds'], metrics, predictions)
    print(json.dumps({key: value for key, value in metrics.items() if key not in ('history', 'provenance')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
