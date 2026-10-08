"""Offline outcome inference from the verified numeric sequence checkpoint."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def success_probabilities(data, configuration, weights_directory, device='auto'):
    # Torch is optional for the saved CPU-only v5; import only when requested.
    import torch
    from referee.data import CODE_HASH
    from referee.relations import RELATIONS_HASH
    from referee.sequence import SequenceNetwork, batch_arrays, signed_log

    path = Path(weights_directory)/configuration['checkpoint']
    if configuration.get('sha256'):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != configuration['sha256']:
            raise ValueError('Neural success checkpoint checksum mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['features_code_hash'] != CODE_HASH or checkpoint['relations_hash'] != RELATIONS_HASH:
        raise ValueError('Neural success feature provenance mismatch')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(4)
    stats = checkpoint['normalization']
    run = np.clip((signed_log(data['X'][checkpoint['run_columns']].to_numpy(dtype=np.float32))-
                   stats['run_mean'])/stats['run_std'], -8, 8)
    columns = [data['turn_columns'].index(c) for c in checkpoint['turn_columns']]
    turns = np.clip((signed_log(data['turn_X'][:, columns])-stats['turn_mean'])/stats['turn_std'], -8, 8)
    network = SequenceNetwork(run.shape[1], turns.shape[1], checkpoint['width']).to(device)
    network.load_state_dict(checkpoint['state'])
    network.eval()
    indexes = np.arange(len(data['run_id']))
    values = []
    with torch.inference_mode():
        for start in range(0, len(indexes), 64):
            selected = indexes[start:start+64]
            batch = tuple(x.to(device) for x in batch_arrays(run, turns, data['offsets'], selected))
            _, success, _ = network(*batch)
            values.append(success.sigmoid().cpu().numpy())
    return np.concatenate(values)
