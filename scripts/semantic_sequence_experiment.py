"""Warm-start v6 with full frozen semantic vectors; selection is fold1 half only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
from referee.semantic_sequence import (SemanticSequenceNetwork, semantic_sequences,
    concatenate_semantics, normalize_checkpoint, semantic_batch_arrays, semantic_predict,
    SEMANTIC_SEQUENCE_HASH)
from scripts.experiment import report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='semantic_sequence_v14')
    ap.add_argument('--init', default='outputs/sequence_v6/model.pt')
    ap.add_argument('--train', default='data/competition/train.csv')
    ap.add_argument('--extra-train', default='data/train.csv')
    ap.add_argument('--encoder', default='weights/encoder')
    ap.add_argument('--cache', default='outputs/cache')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--learning-rate', type=float, default=6e-5)
    ap.add_argument('--semantic-lr', type=float, default=1e-3)
    ap.add_argument('--ema-decay', type=float, default=.995)
    ap.add_argument('--semantic-width', type=int, default=64)
    ap.add_argument('--no-goal-difference', action='store_true')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--encoder-device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--encode-only', action='store_true')
    args = ap.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4); torch.manual_seed(44)
    if args.epochs < 0 or not 0 <= args.ema_decay < 1:
        raise ValueError('epochs must be nonnegative and EMA decay in [0,1)')
    if (args.device == 'cuda' or args.encoder_device == 'cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device(args.device); rng = np.random.default_rng(44)
    checkpoint = torch.load(args.init, map_location='cpu', weights_only=False)
    if checkpoint['features_code_hash'] != CODE_HASH or checkpoint['relations_hash'] != RELATIONS_HASH:
        raise ValueError('Numeric checkpoint provenance mismatch')
    data = relation_features(load_data(args.train), args.train)
    folds = fixed_folds(data); valid = np.flatnonzero(folds == 1); reserved = np.flatnonzero(folds == 0)
    training = np.flatnonzero(folds > 1)
    semantics = semantic_sequences(data, args.encoder, args.cache, device=args.encoder_device)
    excluded = set(data['groups'][folds < 2])
    if args.extra_train:
        extra = relation_features(load_data(args.extra_train), args.extra_train)
        extra_semantics = semantic_sequences(extra, args.encoder, args.cache, device=args.encoder_device)
        selected = np.array([i for i,g in enumerate(extra['groups']) if g not in excluded])+len(data['y'])
        training = np.concatenate([training, selected]); data = concatenate(data, extra)
        semantics = concatenate_semantics(semantics, extra_semantics)
    if args.encode_only:
        print(json.dumps({'rows': len(data['y']), 'unique_vector_rows': len(semantics['vectors']),
                          'semantic_metadata': semantics['metadata'], 'seconds': time.monotonic()-started}, indent=2), flush=True)
        return
    if len(training) != 16000:
        raise ValueError('This v6 warm-start experiment expects the original 16000 training rows')
    assert not (set(data['groups'][training]) & set(data['groups'][np.concatenate([valid, reserved])]))
    select, check = train_test_split(valid, test_size=.5, stratify=data['y'][valid], random_state=12)
    run, turns = normalize_checkpoint(data, checkpoint)
    network = SemanticSequenceNetwork(run.shape[1], turns.shape[1], checkpoint['width'],
                                      args.semantic_width, not args.no_goal_difference).to(device)
    network.warm_start(checkpoint['state'])
    averaged = copy.deepcopy(network).eval()
    averaged.requires_grad_(False)
    base_parameters = [parameter for name,parameter in network.named_parameters()
                       if not name.startswith('semantic_')]
    semantic_parameters = [parameter for name,parameter in network.named_parameters()
                           if name.startswith('semantic_')]
    optimizer = torch.optim.AdamW([
        {'params':base_parameters, 'lr':args.learning_rate},
        {'params':semantic_parameters, 'lr':args.semantic_lr}], weight_decay=.02)
    # Relative cosine keeps a 10% floor for each group's own initial rate.
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: .1+.9*(1+math.cos(math.pi*min(epoch,args.epochs)/max(args.epochs,1)))/2)
    folder = ROOT/'outputs'/args.name; folder.mkdir(parents=True, exist_ok=True)
    best = -1.; history = []
    normalization_digest = hashlib.sha256()
    for key, value in sorted(checkpoint['normalization'].items()):
        normalization_digest.update(key.encode('utf8'))
        normalization_digest.update(np.asarray(value).tobytes())
    provenance = {'encoder_sha256': semantics['metadata']['encoder_sha256'],
                  'semantic_code_sha256': SEMANTIC_SEQUENCE_HASH,
                  'normalization_sha256': normalization_digest.hexdigest(),
                  'normalization_source': 'Frozen numerical training-only moments from initialization checkpoint',
                  'semantic_schema': {'embedding_dimensions':384, 'semantic_width':args.semantic_width,
                                      'goal_difference':not args.no_goal_difference, 'vector_normalization':'L2'},
                  'semantic_sources': semantics['metadata'].get('sources', [semantics['metadata']]),
                  'initialization_sha256': hashlib.sha256(Path(args.init).read_bytes()).hexdigest(),
                  'training_groups_sha256': hashlib.sha256('\n'.join(sorted(data['groups'][training])).encode()).hexdigest(),
                  'experiment_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for epoch in range(args.epochs+1):
        epoch_started = time.monotonic(); losses = []
        if epoch:
            network.train(); order = rng.permutation(training)
            for start in range(0, len(order), args.batch_size):
                selected = order[start:start+args.batch_size]
                batch = tuple(x.to(device) for x in semantic_batch_arrays(run, turns, data['offsets'], selected, semantics))
                yn = data['y'][selected].astype(np.int64); y = torch.from_numpy(yn).to(device)
                success = torch.from_numpy(data['success'][selected].astype(np.float32)).to(device)
                labels, ps, onset = network(*batch)
                loss = torch.nn.functional.cross_entropy(labels,y)+.5*torch.nn.functional.binary_cross_entropy_with_logits(ps,success)
                faulty = y > 0
                if faulty.any():
                    local = onset[faulty,:,y[faulty]-1]
                    target = torch.from_numpy(data['fault'][selected][yn>0].astype(np.int64)).to(device)
                    loss = loss+.3*torch.nn.functional.cross_entropy(local,target)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(),2); optimizer.step()
                with torch.no_grad():
                    for average,current in zip(averaged.parameters(),network.parameters()):
                        average.lerp_(current,1-args.ema_decay)
                losses.append(float(loss.detach()))
            schedule.step()
        p, ps, candidates = semantic_predict(averaged,run,turns,data['offsets'],select,semantics,args.batch_size)
        labels=p.argmax(1); onset=candidates[np.arange(len(select)),labels]
        metrics=report(data,select,labels,(ps>=.45).astype(int),onset,training)
        entry={'epoch':epoch,'loss':float(np.mean(losses)) if losses else None,
               'seconds':round(time.monotonic()-epoch_started,2),
               'learning_rate':optimizer.param_groups[0]['lr'],
               'semantic_learning_rate':optimizer.param_groups[1]['lr'],
               'selection_score_proxy':metrics['score_proxy']}
        history.append(entry);print(json.dumps(entry),flush=True)
        if metrics['score_proxy'] > best:
            best=metrics['score_proxy'];result=dict(checkpoint)
            result.update(state={key:value.detach().cpu() for key,value in averaged.state_dict().items()},
                          epoch=epoch, semantic_provenance=provenance, configuration=vars(args))
            torch.save(result,folder/'model.pt')
    chosen=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    averaged.load_state_dict(chosen['state'])
    # Check-half and fold zero are evaluated only after the selected epoch freezes.
    p,ps,candidates=semantic_predict(averaged,run,turns,data['offsets'],valid,semantics,args.batch_size)
    labels=p.argmax(1);success=(ps>=.45).astype(int);onset=candidates[np.arange(len(valid)),labels]
    metrics=report(data,valid,labels,success,onset,training)
    for title,indexes in (('selection',select),('check',check)):
        rows=np.flatnonzero(np.isin(valid,indexes))
        metrics[title]=report(data,valid[rows],labels[rows],success[rows],onset[rows],training)
    metrics.update(configuration=vars(args),selected_epoch=chosen['epoch'],history=history,
                   elapsed_seconds=time.monotonic()-started,training_rows=len(training),semantic_provenance=provenance,
                   note='Robustness uses an approximate local cohort; fold zero was previously opened in earlier research.')
    np.savez_compressed(folder/'validation.npz',idx=valid,label_probabilities=p,success_probabilities=ps,
                        labels=labels,success=success,turns=onset,candidates=candidates,folds=folds)
    p0,ps0,candidates0=semantic_predict(averaged,run,turns,data['offsets'],reserved,semantics,args.batch_size)
    labels0=p0.argmax(1);success0=(ps0>=.45).astype(int);onset0=candidates0[np.arange(len(reserved)),labels0]
    metrics['fold0']=report(data,reserved,labels0,success0,onset0,training)
    fold0=folder/'fold0';fold0.mkdir(exist_ok=True)
    (fold0/'metrics.json').write_text(json.dumps(metrics['fold0'],indent=2),encoding='utf8')
    np.savez_compressed(fold0/'validation.npz',idx=reserved,label_probabilities=p0,success_probabilities=ps0,
                        labels=labels0,success=success0,turns=onset0,candidates=candidates0,folds=folds)
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    print(json.dumps({key:value for key,value in metrics.items() if key!='history'},indent=2),flush=True)


if __name__=='__main__':
    main()
