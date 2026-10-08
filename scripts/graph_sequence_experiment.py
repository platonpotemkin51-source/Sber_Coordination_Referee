"""Warm-start v6 with observed graph relations; selection excludes fold 0."""
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
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from referee.data import load_data, fixed_folds, concatenate, CODE_HASH
from referee.relations import relation_features, RELATIONS_HASH
from referee.sequence import normalize_training, signed_log
from referee.graph_sequence import (
    GRAPH_HASH, RELATION_NAMES, GraphNetwork, load_identities,
    graph_batch_arrays, graph_predict,
)
from scripts.experiment import report


def cpu_state(network):
    return {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}


def save_validation(folder, data, valid, training, folds, network, run, turns, identities, batch_size, note):
    p, ps, candidates = graph_predict(network, run, turns, data["offsets"], identities, valid, batch_size)
    labels = p.argmax(1)
    success = (ps >= .45).astype(int)
    onset = candidates[np.arange(len(valid)), labels]
    metrics = report(data, valid, labels, success, onset, training)
    metrics["note"] = note
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "validation.npz", idx=valid, label_probabilities=p,
                        success_probabilities=ps, labels=labels, success=success,
                        turns=onset, candidates=candidates, folds=folds)
    return metrics, (labels, success, onset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="graph_sequence_v13")
    parser.add_argument("--train", default="data/competition/train.csv")
    parser.add_argument("--extra-train", default="data/train.csv")
    parser.add_argument("--warmstart", default="outputs/sequence_v6/model.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--base-lr", type=float, default=6e-5)
    parser.add_argument("--edge-lr", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=0)
    args = parser.parse_args()
    if args.epochs < 0 or not 0 <= args.ema_decay < 1:
        raise ValueError("epochs must be nonnegative and EMA decay must be in [0, 1)")
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    warm_path = Path(args.warmstart)
    warm = torch.load(warm_path, map_location="cpu", weights_only=False)
    if warm["features_code_hash"] != CODE_HASH or warm["relations_hash"] != RELATIONS_HASH:
        raise ValueError("Warmstart features do not match the current v6 extractor")
    data = relation_features(load_data(args.train), args.train)
    folds = fixed_folds(data)
    valid = np.flatnonzero(folds == 1)
    fold0 = np.flatnonzero(folds == 0)
    training = np.flatnonzero(folds > 1)
    sources = [data["source_sha256"]]
    identities = load_identities(args.train, data["offsets"])["identities"]
    if args.extra_train:
        extra = relation_features(load_data(args.extra_train), args.extra_train)
        excluded = set(data["groups"][folds < 2])
        selected = np.array([i for i, group in enumerate(extra["groups"]) if group not in excluded]) + len(data["y"])
        training = np.concatenate([training, selected])
        sources.append(extra["source_sha256"])
        identities = np.concatenate([identities, load_identities(args.extra_train, extra["offsets"])["identities"]])
        data = concatenate(data, extra)
    assert not set(data["groups"][training]) & set(data["groups"][np.concatenate([valid, fold0])])
    if list(data["X"].columns) != warm["run_columns"] or data["turn_columns"] != warm["turn_columns"]:
        raise ValueError("Warmstart feature order differs; avoid silently changing the input")
    # Recompute only to verify the warm checkpoint belongs to this exact training
    # population; use its existing normalizer for all subsequent predictions.
    normalized_run, normalized_turns, actual_stats = normalize_training(data, training)
    del normalized_run, normalized_turns
    stats = warm["normalization"]
    for key, value in actual_stats.items():
        np.testing.assert_allclose(value, stats[key], rtol=1e-6, atol=1e-6)
    run = np.clip((signed_log(data["X"].to_numpy(dtype=np.float32)) - stats["run_mean"]) / stats["run_std"], -8, 8)
    turns = np.clip((signed_log(data["turn_X"]) - stats["turn_mean"]) / stats["turn_std"], -8, 8)
    select, check = train_test_split(valid, test_size=.5, stratify=data["y"][valid], random_state=12)
    network = GraphNetwork(run.shape[1], turns.shape[1], warm["width"]).to(device)
    result = network.load_state_dict(warm["state"], strict=False)
    if result.missing_keys != ["edge_weights"] or result.unexpected_keys:
        raise ValueError(f"Unexpected warmstart state mismatch: {result}")
    base_parameters = [parameter for name, parameter in network.named_parameters() if name != "edge_weights"]
    optimizer = torch.optim.AdamW([
        {"params": base_parameters, "lr": args.base_lr, "weight_decay": .02},
        {"params": [network.edge_weights], "lr": args.edge_lr, "weight_decay": 0},
    ])
    steps = max(1, math.ceil(len(training) / args.batch_size) * args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .1 + .9 * (1 + math.cos(math.pi * min(step, steps) / steps)) / 2)
    averaged = copy.deepcopy(network).eval() if args.ema_decay else None
    folder = ROOT / "outputs" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    certificate = {
        "source_sha256": sources, "features_code_hash": CODE_HASH,
        "relations_hash": RELATIONS_HASH, "graph_hash": GRAPH_HASH,
        "training_groups_sha256": hashlib.sha256("\n".join(data["groups"][training]).encode()).hexdigest(),
        "training_rows": len(training), "excluded_outer_folds": [0, 1],
        "warmstart_sha256": hashlib.sha256(warm_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    best = -1
    history = []
    for epoch in range(args.epochs + 1):
        epoch_started = time.monotonic()
        losses = []
        if epoch:
            network.train()
            order = rng.permutation(training)
            for start in range(0, len(order), args.batch_size):
                selected = order[start:start + args.batch_size]
                batch = tuple(value.to(device) for value in graph_batch_arrays(run, turns, data["offsets"], identities, selected))
                selected_y = data["y"][selected].astype(np.int64)
                y = torch.from_numpy(selected_y).to(device)
                success_target = torch.from_numpy(data["success"][selected].astype(np.float32)).to(device)
                labels, ps, onset = network(*batch)
                loss = torch.nn.functional.cross_entropy(labels, y) + .5 * torch.nn.functional.binary_cross_entropy_with_logits(ps, success_target)
                faulty = y > 0
                if faulty.any():
                    local = onset[faulty, :, y[faulty] - 1]
                    target = torch.from_numpy(data["fault"][selected][selected_y > 0].astype(np.int64)).to(device)
                    loss = loss + .3 * torch.nn.functional.cross_entropy(local, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 2)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach()))
                if averaged is not None:
                    with torch.no_grad():
                        for target_parameter, current in zip(averaged.parameters(), network.parameters()):
                            target_parameter.lerp_(current, 1 - args.ema_decay)
        evaluated = averaged if averaged is not None else network
        p, ps, candidates = graph_predict(evaluated, run, turns, data["offsets"], identities, select, args.batch_size)
        labels = p.argmax(1)
        onset = candidates[np.arange(len(select)), labels]
        metrics = report(data, select, labels, (ps >= .45).astype(int), onset, training)
        log = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else None,
               "seconds": round(time.monotonic() - epoch_started, 2),
               "selection_score_proxy": metrics["score_proxy"],
               "edge_weight_abs_max": float(evaluated.edge_weights.detach().abs().max())}
        history.append(log)
        print(json.dumps(log), flush=True)
        if metrics["score_proxy"] > best:
            best = metrics["score_proxy"]
            checkpoint = {"state": cpu_state(evaluated), "normalization": stats,
                          "run_columns": list(data["X"].columns), "turn_columns": data["turn_columns"],
                          "width": warm["width"], "epoch": epoch, "features_code_hash": CODE_HASH,
                          "relations_hash": RELATIONS_HASH, "graph_hash": GRAPH_HASH,
                          "relation_names": RELATION_NAMES, "certificate": certificate}
            torch.save(checkpoint, folder / "model.pt")

    checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
    network.load_state_dict(checkpoint["state"])
    metrics, predictions = save_validation(folder, data, valid, training, folds, network, run, turns, identities, args.batch_size,
        "Development fold 1; only its selection half chose epoch 0 through the requested epoch budget.")
    check_positions = np.flatnonzero(np.isin(valid, check))
    labels, success, onset = predictions
    metrics["check"] = report(data, valid[check_positions], labels[check_positions], success[check_positions], onset[check_positions], training)
    metrics.update(configuration=vars(args), selected_epoch=checkpoint["epoch"], history=history,
                   elapsed_seconds=time.monotonic() - started, certificate=certificate)
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "history"}, indent=2), flush=True)
    # This opened control is evaluated only after the checkpoint is frozen.
    control, _ = save_validation(folder / "fold0", data, fold0, training, folds, network, run, turns, identities, args.batch_size,
        "Frozen checkpoint evaluation. Fold 0 was previously opened; it is a repeated control, not a fresh independent holdout.")
    control.update(checkpoint_epoch=checkpoint["epoch"], graph_hash=GRAPH_HASH, certificate=certificate)
    (folder / "fold0" / "metrics.json").write_text(json.dumps(control, indent=2), encoding="utf8")
    print(json.dumps({"fold0": control}, indent=2), flush=True)


if __name__ == "__main__":
    main()
