"""Graph edges encode equality, isolate runs, and preserve the v6 warmstart."""
import numpy as np
import torch

from referee.sequence import SequenceNetwork, batch_arrays
from referee.graph_sequence import (
    IDENTITY_COLUMNS, GraphNetwork, graph_batch_arrays, identity_rows, relation_channels,
)


def example_run():
    return {
        "goal": "Verify budget", "agents": [{"id": "a1", "tools": []}, {"id": "a2", "tools": []}],
        "messages": [
            {"t": 0, "from": "a1", "to": "a2", "intent": "check_budget", "type": "handoff", "text": "Assign check_budget.", "refs": []},
            {"t": 1, "from": "a2", "to": "a1", "intent": "", "type": "inform", "text": "Acknowledged, starting.", "refs": []},
            {"t": 2, "from": "a2", "to": "a1", "intent": "", "type": "handoff", "text": "Finished.", "refs": ["art_1"]},
            {"t": 3, "from": "a1", "to": "a2", "intent": "", "type": "state_update", "text": "Shared value updated.", "refs": ["art_1"]},
        ],
        "artifacts": [{"id": "art_1", "t": 2, "by": "a2", "subtask": "check_budget"}],
        "shared_state": [{"key": "cost_total", "t": 2}, {"key": "cost_total", "t": 3}],
    }


def test_identifier_permutation_preserves_all_relations():
    values = identity_rows(example_run())
    changed = values.copy()
    # Agent sender/recipient share one namespace; other namespaces are separate.
    for columns in ([0, 1], [2], [3, 4, 5, 6], [7, 8, 9, 10]):
        original = values[:, columns]
        changed[:, columns] = np.where(original == 0, 0, 97 - 7 * original)
    padding = torch.zeros((1, len(values)), dtype=torch.bool)
    original_relations = relation_channels(torch.from_numpy(values[None]), padding)
    changed_relations = relation_channels(torch.from_numpy(changed[None]), padding)
    torch.testing.assert_close(original_relations, changed_relations, rtol=0, atol=0)
    # The missing intent was resolved from a pending assignment and a ref.
    assert values[0, 2] == values[1, 2] == values[2, 2] != 0
    assert original_relations[0, 3, 2, 3]
    assert original_relations[0, 4, 2, 3]


def test_unknown_identities_do_not_create_edges():
    values = torch.zeros((1, 5, len(IDENTITY_COLUMNS)), dtype=torch.int32)
    padding = torch.tensor([[False, False, False, False, True]])
    relations = relation_channels(values, padding)
    assert not relations[:, :5].any()
    assert relations[0, 5, 0, 3]
    assert not relations[0, 5, 0, 4]
    assert not relations[:, :, 4].any()
    assert not relations[:, :, :, 4].any()


def test_relations_are_isolated_between_runs_even_when_local_ids_match():
    first = identity_rows(example_run())
    second = np.zeros_like(first)
    second[:2, :3] = [[1, 2, 1], [2, 1, 2]]
    batch = torch.from_numpy(np.stack([first, second]))
    padding = torch.zeros((2, len(first)), dtype=torch.bool)
    together = relation_channels(batch, padding)
    separate = relation_channels(batch[:1], padding[:1])
    torch.testing.assert_close(together[:1], separate, rtol=0, atol=0)
    batch[1] = 42
    torch.testing.assert_close(relation_channels(batch, padding)[:1], separate, rtol=0, atol=0)


def test_zero_bias_eval_matches_v6_exactly_and_padding_remains_masked():
    torch.manual_seed(7)
    original = SequenceNetwork(2, 3, 24).eval()
    graph = GraphNetwork(2, 3, 24).eval()
    result = graph.load_state_dict(original.state_dict(), strict=False)
    assert result.missing_keys == ["edge_weights"]
    run = np.arange(4, dtype=np.float32).reshape(2, 2)
    turns = np.arange(18, dtype=np.float32).reshape(6, 3)
    offsets = np.array([0, 2, 6])
    values = np.zeros((6, len(IDENTITY_COLUMNS)), dtype=np.int32)
    values[:, 0] = 1
    indexes = np.array([0, 1])
    with torch.inference_mode():
        reference = original(*batch_arrays(run, turns, offsets, indexes))
        observed = graph(*graph_batch_arrays(run, turns, offsets, values, indexes))
    for wanted, actual in zip(reference, observed):
        torch.testing.assert_close(wanted, actual, rtol=0, atol=0)
    assert torch.all(observed[2][0, 2:] == -1e4)


def test_zero_bias_training_has_finite_edge_gradients_and_nonzero_path():
    torch.manual_seed(8)
    network = GraphNetwork(2, 3, 24).train()
    run = np.ones((2, 2), dtype=np.float32)
    turns = np.arange(18, dtype=np.float32).reshape(6, 3) / 10
    offsets = np.array([0, 2, 6])
    values = np.zeros((6, len(IDENTITY_COLUMNS)), dtype=np.int32)
    values[:, 0] = [1, 2, 1, 2, 1, 1]
    values[:, 1] = [2, 1, 2, 1, 2, 2]
    values[:, 2] = [1, 1, 1, 2, 1, 2]
    batch = graph_batch_arrays(run, turns, offsets, values, np.array([0, 1]))
    labels, success, onset = network(*batch)
    loss = torch.nn.functional.cross_entropy(labels, torch.tensor([1, 2])) + success.square().mean()
    loss.backward()
    assert network.edge_weights.grad is not None
    assert torch.isfinite(network.edge_weights.grad).all()
    assert network.edge_weights.grad.abs().max() > 0
    with torch.no_grad():
        network.edge_weights.add_(.1)
    network.eval()
    with torch.inference_mode():
        outputs = network(*batch)
    assert all(torch.isfinite(value).all() for value in outputs)
    assert torch.all(outputs[2][0, 2:] == -1e4)


def test_eval_preserves_magnitude_of_additive_attention_bias():
    # Regression for torch 2.5's fused encoder converting float biases to bool.
    torch.manual_seed(42)
    network = GraphNetwork(2, 3, 24).eval()
    run = np.random.default_rng(1).normal(size=(2, 2)).astype(np.float32)
    turns = np.random.default_rng(2).normal(size=(12, 3)).astype(np.float32)
    offsets = np.array([0, 5, 12])
    identities = np.zeros((12, len(IDENTITY_COLUMNS)), dtype=np.int32)
    identities[:, 0] = [1, 2, 1, 3, 1, 1, 2, 1, 2, 3, 1, 2]
    identities[:, 1] = 2
    identities[:, 2] = [1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1]
    batch = graph_batch_arrays(run, turns, offsets, identities, np.array([0, 1]))
    previous = torch.backends.mha.get_fastpath_enabled()
    outputs = []
    with torch.inference_mode():
        for magnitude in (.05, .2):
            network.edge_weights.zero_()
            network.edge_weights[:, :, 1].fill_(magnitude)
            outputs.append(network(*batch)[0])
    assert torch.backends.mha.get_fastpath_enabled() == previous
    assert (outputs[0] - outputs[1]).abs().max() > 1e-5
