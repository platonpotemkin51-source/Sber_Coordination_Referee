"""Relation-biased sequence model; identities have meaning only within a run."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from baseline.features import parse_run
from referee.events import GENERIC, GROUPS, SNAKE
from referee.sequence import SequenceNetwork, batch_arrays

GRAPH_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
RELATION_NAMES = (
    "same_agent", "same_task", "communication", "same_artifact",
    "same_state", "time_neighborhood",
)
IDENTITY_COLUMNS = (
    "sender", "recipient", "task", "artifact_0", "artifact_1",
    "artifact_2", "artifact_3", "state_0", "state_1", "state_2", "state_3",
)


def identity_rows(run):
    """Encode observed equalities without labels or persistent identifier values.

    Zero denotes missing identity. Four artifact/state slots retain multiple
    links while keeping a compact fixed-width array rather than pairwise edges.
    Artifact creation is linked to the sender's message at the same timestamp.
    """
    messages = sorted(run["messages"], key=lambda m: m["t"])
    artifacts = run.get("artifacts", [])
    states = run.get("shared_state", [])
    by_artifact = {a["id"]: a for a in artifacts}
    excluded = {s.get("key", "") for s in states}
    excluded.update(tool for agent in run.get("agents", []) for tool in agent.get("tools", []))
    known = {m.get("intent") for m in messages} | {a.get("subtask") for a in artifacts}
    known.difference_update(GENERIC | {None})
    tasks, assignment, delivery = [], [], []
    for message in messages:
        text = message.get("text", "").lower()
        tokens = [x for x in SNAKE.findall(text) if x not in excluded and not x.startswith("art_") and x not in GENERIC]
        tokens.sort(key=lambda x: x not in known)
        task = message.get("intent", "") or ""
        if task in GENERIC:
            task = tokens[0] if tokens else ""
        linked = [by_artifact[r] for r in message.get("refs", []) if r in by_artifact]
        if not task:
            task = next((a.get("subtask") for a in linked if a.get("subtask") not in GENERIC | {None}), "")
        is_delivery = message.get("type") == "handoff" and bool(
            message.get("refs") or linked or "art_" in text
            or any(word in text for word in GROUPS["result"])
        )
        tasks.append(task if task not in GENERIC else "")
        delivery.append(is_delivery)
        assignment.append(message.get("type") == "handoff" and not is_delivery)

    pending = defaultdict(set)
    for index, message in enumerate(messages):
        if assignment[index] and tasks[index]:
            pending[message.get("to")].add(tasks[index])
        if not tasks[index] and not message.get("intent") and len(pending[message.get("from")]) == 1:
            tasks[index] = next(iter(pending[message.get("from")]))
        if delivery[index] and tasks[index]:
            pending[message.get("from")].discard(tasks[index])

    def encoder():
        mapping = {}

        def identify(value):
            if value is None or value == "":
                return 0
            if value not in mapping:
                mapping[value] = len(mapping) + 1
            return mapping[value]

        return identify

    agent_id, task_id, artifact_id, state_id = (encoder() for _ in range(4))
    created = defaultdict(list)
    states_at = defaultdict(list)
    for artifact in artifacts:
        created[(artifact.get("t", -1), artifact.get("by"))].append(artifact["id"])
    state_patterns = []
    state_keys = list(dict.fromkeys(s.get("key") for s in states if s.get("key")))
    for key in state_keys:
        state_patterns.append((key, re.compile(r"(?<!\w)" + re.escape(str(key)) + r"(?!\w)", re.IGNORECASE)))
    for state in states:
        if state.get("key"):
            states_at[state.get("t", -1)].append(state["key"])
    result = np.zeros((len(messages), len(IDENTITY_COLUMNS)), dtype=np.int32)
    for index, message in enumerate(messages):
        result[index, :3] = [agent_id(message.get("from")), agent_id(message.get("to")), task_id(tasks[index])]
        text = message.get("text", "")
        linked_artifacts = [r for r in message.get("refs", []) if r in by_artifact]
        linked_artifacts += [r for r in re.findall(r"\bart_[\w]+\b", text) if r in by_artifact]
        linked_artifacts += created[(message["t"], message.get("from"))]
        # Preserve encounter order, so renaming identifiers does not reorder slots.
        linked_artifacts = list(dict.fromkeys(linked_artifacts))[:4]
        linked_states = list(dict.fromkeys(
            states_at[message["t"]] + [key for key, pattern in state_patterns if pattern.search(text)]
        ))[:4]
        for slot, value in enumerate(linked_artifacts):
            result[index, 3 + slot] = artifact_id(value)
        for slot, value in enumerate(linked_states):
            result[index, 7 + slot] = state_id(value)
    return result


def load_identities(source, expected_offsets=None, cache="outputs/cache"):
    """Cache only int32 per-turn identity slots, keyed by source and extractor."""
    source = Path(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = Path(cache) / f"graph-identities-{digest[:16]}-{GRAPH_HASH}.joblib"
    if target.exists():
        result = joblib.load(target)
    else:
        records = pd.read_csv(source).to_dict("records")
        blocks, offsets = [], [0]
        for record in records:
            block = identity_rows(parse_run(record))
            blocks.append(block)
            offsets.append(offsets[-1] + len(block))
        result = {
            "identities": np.concatenate(blocks), "offsets": np.array(offsets),
            "source_sha256": digest, "graph_hash": GRAPH_HASH,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, target, compress=3)
    if expected_offsets is not None:
        np.testing.assert_array_equal(result["offsets"], expected_offsets)
    return result


def _same_known(values):
    return (values[:, :, None] == values[:, None, :]) & (values[:, :, None] != 0)


def _share_known(values):
    batch, length, slots = values.shape
    result = torch.zeros((batch, length, length), dtype=torch.bool, device=values.device)
    for slot in range(slots):
        current = values[:, :, slot]
        result |= ((current[:, :, None, None] == values[:, None, :, :]) & (current[:, :, None, None] != 0)).any(-1)
    return result


def relation_channels(identities, padding):
    """Return [batch, six channels, turns, turns]; batches never share edges."""
    if identities.shape[:2] != padding.shape or identities.shape[-1] != len(IDENTITY_COLUMNS):
        raise ValueError("Identity tensor must match the padded turn tensor")
    sender, recipient, task = (identities[:, :, index] for index in range(3))
    communication = (recipient[:, :, None] == sender[:, None, :]) & (recipient[:, :, None] != 0)
    positions = torch.arange(identities.shape[1], device=identities.device)
    neighborhood = (positions[:, None] - positions[None, :]).abs() <= 3
    channels = torch.stack([
        _same_known(sender), _same_known(task), communication,
        _share_known(identities[:, :, 3:7]), _share_known(identities[:, :, 7:11]),
        neighborhood[None].expand(identities.shape[0], -1, -1),
    ], dim=1)
    valid = (~padding)[:, None, :, None] & (~padding)[:, None, None, :]
    return channels & valid


class GraphNetwork(SequenceNetwork):
    """Warm-compatible v6 model with additive, learned relation attention bias."""

    def __init__(self, run_features, turn_features, width=96):
        super().__init__(run_features, turn_features, width)
        heads = self.sequence.layers[0].self_attn.num_heads
        self.edge_weights = nn.Parameter(torch.zeros(len(self.sequence.layers), heads, len(RELATION_NAMES)))

    def forward(self, run, turns, padding, identities):
        # The unchanged v6 path guarantees exact equality for the epoch-0 eval.
        # Training always uses edges, including at zero, to preserve gradients.
        if not self.training and torch.count_nonzero(self.edge_weights.detach()).item() == 0:
            return super().forward(run, turns, padding)
        global_token = self.run_encoder(run)
        tokens = self.turn_encoder(turns)
        position = torch.arange(tokens.shape[1], device=tokens.device, dtype=tokens.dtype)[None, :, None]
        dimensions = torch.arange(tokens.shape[2], device=tokens.device, dtype=tokens.dtype)[None, None, :]
        positional = torch.sin(position / torch.pow(10000, 2 * torch.floor(dimensions / 2) / tokens.shape[2]) + (dimensions % 2) * torch.pi / 2)
        tokens = tokens + .1 * positional
        encoded = torch.cat([global_token[:, None], tokens], dim=1)
        full_padding = torch.cat([torch.zeros((len(run), 1), device=run.device, dtype=torch.bool), padding], dim=1)
        # Masks have one float dtype, avoiding the bool/float mismatch in 2.5.
        key_padding = torch.zeros_like(full_padding, dtype=encoded.dtype).masked_fill(full_padding, -torch.inf)
        channels = relation_channels(identities, padding).to(encoded.dtype)
        # The fused encoder inference path in torch 2.5 treats arbitrary
        # float attention biases like Boolean masks. This discards magnitude
        # and sign, making learned relation weights ineffective at evaluation.
        # The standard scaled-dot-product path preserves additive float bias.
        fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            for layer_index, layer in enumerate(self.sequence.layers):
                local_bias = torch.einsum("bcij,hc->bhij", channels, self.edge_weights[layer_index].to(encoded.dtype))
                bias = encoded.new_zeros((len(run), local_bias.shape[1], tokens.shape[1] + 1, tokens.shape[1] + 1))
                bias[:, :, 1:, 1:] = local_bias
                encoded = layer(encoded, src_mask=bias.flatten(0, 1), src_key_padding_mask=key_padding)
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)
        if self.sequence.norm is not None:
            encoded = self.sequence.norm(encoded)
        local = encoded[:, 1:]
        valid = (~padding)[:, :, None]
        mean = (local * valid).sum(1) / valid.sum(1).clamp(min=1)
        maximum = local.masked_fill(padding[:, :, None], -1e4).amax(1)
        fused = torch.cat([encoded[:, 0], mean, maximum], dim=1)
        return self.classifier(fused), self.success(fused)[:, 0], self.onset(local).masked_fill(padding[:, :, None], -1e4)


def graph_batch_arrays(run, turns, offsets, identities, indexes):
    batch = batch_arrays(run, turns, offsets, indexes)
    packed = np.zeros((len(indexes), batch[1].shape[1], len(IDENTITY_COLUMNS)), dtype=np.int32)
    for row, index in enumerate(indexes):
        block = identities[offsets[index]:offsets[index + 1]]
        packed[row, :len(block)] = block
    return (*batch, torch.from_numpy(packed))


def graph_predict(network, run, turns, offsets, identities, indexes, batch_size=64):
    network.eval()
    device = next(network.parameters()).device
    probabilities, successes, candidates = [], [], []
    with torch.inference_mode():
        for start in range(0, len(indexes), batch_size):
            selected = indexes[start:start + batch_size]
            batch = tuple(value.to(device) for value in graph_batch_arrays(run, turns, offsets, identities, selected))
            labels, ps, onset = network(*batch)
            probabilities.append(labels.softmax(-1).cpu().numpy())
            successes.append(ps.sigmoid().cpu().numpy())
            onset = onset.softmax(1).cpu().numpy()
            for row, index in enumerate(selected):
                length = int(offsets[index + 1] - offsets[index])
                scores = onset[row, :length]
                sums = np.array([scores[max(0, turn - 2):min(length, turn + 3)].sum(0) for turn in range(length)])
                candidates.append([-1, *sums.argmax(0).tolist()])
    return np.concatenate(probabilities), np.concatenate(successes), np.array(candidates)
