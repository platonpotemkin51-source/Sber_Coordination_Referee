"""Serialized outcome inference preserves schema alignment and ignores targets."""
import numpy as np
import pandas as pd
import pytest
import torch

from referee.data import CODE_HASH
from referee.relations import RELATIONS_HASH
from referee.sequence import SequenceNetwork
from referee.neural_success import success_probabilities


def test_checkpoint_inference_ignores_targets_and_column_order(tmp_path):
    torch.manual_seed(21)
    network = SequenceNetwork(2, 3, 24)
    checkpoint = {'state': network.state_dict(), 'width': 24,
                  'features_code_hash': CODE_HASH, 'relations_hash': RELATIONS_HASH,
                  'run_columns': ['x', 'z'], 'turn_columns': ['a', 'b', 'c'],
                  'normalization': {'run_mean': np.zeros(2, np.float32), 'run_std': np.ones(2, np.float32),
                                    'turn_mean': np.zeros(3, np.float32), 'turn_std': np.ones(3, np.float32)}}
    torch.save(checkpoint, tmp_path/'success.pt')
    data = {'X': pd.DataFrame({'x':[1., 2.], 'z':[3., 4.]}),
            'turn_X': np.arange(15, dtype=np.float32).reshape(5, 3),
            'turn_columns': ['a', 'b', 'c'], 'offsets': np.array([0, 2, 5]),
            'run_id': np.array(['first', 'second']), 'y': np.array([1, 2]),
            'success': np.array([0, 1]), 'fault': np.array([1, 3])}
    config = {'checkpoint': 'success.pt'}
    reference = success_probabilities(data, config, tmp_path, 'cpu')
    data['X'] = data['X'][['z', 'x']]
    data['turn_X'] = data['turn_X'][:, [2, 0, 1]]
    data['turn_columns'] = ['c', 'a', 'b']
    data['y'][:] = 6; data['success'][:] = 0; data['fault'][:] = 100
    data['run_id'][:] = 'other'
    observed = success_probabilities(data, config, tmp_path, 'cpu')
    np.testing.assert_array_equal(reference, observed)
    assert np.isfinite(observed).all() and ((observed >= 0) & (observed <= 1)).all()
    with pytest.raises(ValueError, match='checksum'):
        success_probabilities(data, {**config, 'sha256': 'incorrect'}, tmp_path, 'cpu')
