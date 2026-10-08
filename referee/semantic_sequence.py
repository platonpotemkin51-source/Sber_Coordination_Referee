"""Frozen full MiniLM vectors with compact indexes and residual sequence inputs."""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from referee.sequence import SequenceNetwork, signed_log, batch_arrays

SEMANTIC_SEQUENCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
EMBEDDING_DIMENSIONS = 384


def encoder_sha256(encoder):
    """Fingerprint the local weights and all tokenizer/model configuration."""
    encoder = Path(encoder)
    weight = encoder/'model.safetensors'
    if not weight.exists():
        weight = encoder/'pytorch_model.bin'
    if not weight.exists():
        raise FileNotFoundError(f'No local encoder weights in {encoder}')
    names = [weight.name, 'config.json', 'tokenizer.json', 'tokenizer_config.json',
             'special_tokens_map.json', 'vocab.txt', 'added_tokens.json']
    digest = hashlib.sha256()
    for name in sorted(names):
        path = encoder/name
        if path.exists():
            digest.update(name.encode('utf8'))
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024*1024), b''):
                    digest.update(block)
    return digest.hexdigest()


def text_index(documents, offsets):
    """Only normalized observable goal/message text enters the vocabulary."""
    texts = []
    for i, document in enumerate(documents):
        lines = document.splitlines()
        if not lines or not lines[0].startswith('GOAL '):
            raise ValueError('Expected normalized document starting with GOAL')
        goal = lines[0].removeprefix('GOAL ').replace('_', ' ')
        messages = [line.split(' ', 1)[-1].replace('_', ' ') for line in lines[1:]]
        if len(messages) != int(offsets[i+1]-offsets[i]):
            raise ValueError('Semantic messages and numeric event offsets differ')
        texts.append((goal, messages))
    corpus = sorted({text for goal, messages in texts for text in [goal, *messages]})
    lookup = {text: i for i, text in enumerate(corpus)}
    goals = np.array([lookup[goal] for goal, _ in texts], dtype=np.int32)
    turns = np.array([lookup[text] for _, messages in texts for text in messages], dtype=np.int32)
    return corpus, goals, turns


def _text_sha(corpus):
    digest = hashlib.sha256()
    for text in corpus:
        value = text.encode('utf8')
        digest.update(len(value).to_bytes(8, 'little'))
        digest.update(value)
    return digest.hexdigest()


def encode_local(corpus, encoder, batch_size=64, device='cpu'):
    """Mean-pool and L2-normalize the frozen local encoder; never download."""
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(encoder), local_files_only=True)
    model = AutoModel.from_pretrained(str(encoder), local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    vectors = []
    print(f'Encoding {len(corpus)} unique semantic texts on {device}', flush=True)
    with torch.inference_mode():
        for start in range(0, len(corpus), batch_size):
            inputs = tokenizer(corpus[start:start+batch_size], padding=True,
                               truncation=True, max_length=96, return_tensors='pt')
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hidden = model(**inputs).last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1)
            pooled = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
            vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
    result = np.concatenate(vectors).astype(np.float32)
    if result.shape != (len(corpus), EMBEDDING_DIMENSIONS):
        raise ValueError(f'Expected MiniLM 384-dimensional vectors, got {result.shape}')
    return result


def semantic_sequences(data, encoder='weights/encoder', cache='outputs/cache',
                       batch_size=64, device='cpu', encode=None):
    """Cache unique vectors plus indexes, not one dense vector per turn.

    Source, encoder, implementation, text and offsets are checked on every load.
    Targets and run identifiers are intentionally never read.
    """
    corpus, goal_indexes, turn_indexes = text_index(data['documents'], data['offsets'])
    metadata = {'source_sha256': data['source_sha256'], 'encoder_sha256': encoder_sha256(encoder),
                'semantic_code_sha256': SEMANTIC_SEQUENCE_HASH, 'corpus_sha256': _text_sha(corpus),
                'goal_indexes_sha256': hashlib.sha256(goal_indexes.tobytes()).hexdigest(),
                'turn_indexes_sha256': hashlib.sha256(turn_indexes.tobytes()).hexdigest(),
                'offsets_sha256': hashlib.sha256(np.asarray(data['offsets'], dtype=np.int64).tobytes()).hexdigest(),
                'embedding_dimensions': EMBEDDING_DIMENSIONS}
    target = Path(cache)/('semantic-sequence-'+metadata['source_sha256'][:16]+'-'+
                         metadata['encoder_sha256'][:16]+'-'+SEMANTIC_SEQUENCE_HASH[:16]+'.joblib')
    if target.exists():
        result = joblib.load(target)
        if result['metadata'] != metadata:
            raise ValueError('Semantic cache provenance differs from input texts/offsets')
        if result['vectors'].shape != (len(corpus), EMBEDDING_DIMENSIONS):
            raise ValueError('Semantic cache shape mismatch')
        np.testing.assert_array_equal(result['goal_indexes'], goal_indexes)
        np.testing.assert_array_equal(result['turn_indexes'], turn_indexes)
        return result
    vectors = (encode or encode_local)(corpus, encoder, batch_size=batch_size, device=device)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(corpus), EMBEDDING_DIMENSIONS) or not np.isfinite(vectors).all():
        raise ValueError('Invalid semantic embedding shape or values')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError('Semantic embedding has zero length')
    vectors = vectors/norms
    result = {'vectors': vectors, 'goal_indexes': goal_indexes, 'turn_indexes': turn_indexes,
              'corpus': corpus, 'metadata': metadata}
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, target, compress=3)
    return result


def concatenate_semantics(first, second):
    for key in ('encoder_sha256', 'semantic_code_sha256', 'embedding_dimensions'):
        if first['metadata'][key] != second['metadata'][key]:
            raise ValueError(f'Incompatible semantic datasets: {key}')
    shift = len(first['vectors'])
    return {'vectors': np.concatenate([first['vectors'], second['vectors']]),
            'goal_indexes': np.concatenate([first['goal_indexes'], second['goal_indexes']+shift]),
            'turn_indexes': np.concatenate([first['turn_indexes'], second['turn_indexes']+shift]),
            'metadata': {**{key:first['metadata'][key] for key in ('encoder_sha256', 'semantic_code_sha256', 'embedding_dimensions')},
                         'sources': [first['metadata'], second['metadata']]}}


def normalize_checkpoint(data, checkpoint):
    """Apply v6 training-only moments without recomputing any normalization."""
    stats = checkpoint['normalization']
    run = np.clip((signed_log(data['X'][checkpoint['run_columns']].to_numpy(dtype=np.float32))-
                   stats['run_mean'])/stats['run_std'], -8, 8)
    columns = [data['turn_columns'].index(c) for c in checkpoint['turn_columns']]
    turns = np.clip((signed_log(data['turn_X'][:, columns])-stats['turn_mean'])/stats['turn_std'], -8, 8)
    return run, turns


class SemanticSequenceNetwork(SequenceNetwork):
    def __init__(self, run_features, turn_features, width=96, semantic_width=64, goal_difference=True):
        super().__init__(run_features, turn_features, width)
        self.goal_difference = goal_difference
        self.semantic_turn = nn.Sequential(nn.Linear(EMBEDDING_DIMENSIONS*(2 if goal_difference else 1), semantic_width),
                                           nn.GELU(), nn.Linear(semantic_width, width))
        self.semantic_goal = nn.Sequential(nn.Linear(EMBEDDING_DIMENSIONS, semantic_width),
                                           nn.GELU(), nn.Linear(semantic_width, width))
        # Exact evaluation warm start, while the final projections receive
        # gradients immediately and propagate upstream after their first update.
        for module in (self.semantic_turn[-1], self.semantic_goal[-1]):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, run, turns, padding, goals, messages):
        global_token = self.run_encoder(run)+self.semantic_goal(goals)
        semantic = torch.cat([messages, messages-goals[:,None]], dim=-1) if self.goal_difference else messages
        tokens = self.turn_encoder(turns)+self.semantic_turn(semantic)
        position = torch.arange(tokens.shape[1], device=tokens.device, dtype=tokens.dtype)[None,:,None]
        dimensions = torch.arange(tokens.shape[2], device=tokens.device, dtype=tokens.dtype)[None,None,:]
        positional = torch.sin(position/torch.pow(10000, 2*torch.floor(dimensions/2)/tokens.shape[2])+(dimensions%2)*torch.pi/2)
        tokens = tokens+.1*positional
        joined = torch.cat([global_token[:,None], tokens], dim=1)
        full_padding = torch.cat([torch.zeros((len(run),1), device=run.device, dtype=torch.bool), padding], dim=1)
        encoded = self.sequence(joined, src_key_padding_mask=full_padding)
        local = encoded[:,1:]
        valid = (~padding)[:,:,None]
        mean = (local*valid).sum(1)/valid.sum(1).clamp(min=1)
        maximum = local.masked_fill(padding[:,:,None], -1e4).amax(1)
        fused = torch.cat([encoded[:,0], mean, maximum], dim=1)
        return self.classifier(fused), self.success(fused)[:,0], self.onset(local).masked_fill(padding[:,:,None], -1e4)

    def warm_start(self, state):
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not key.startswith(('semantic_turn.', 'semantic_goal.')) for key in result.missing_keys):
            raise ValueError(f'Incompatible numeric warm start: {result}')


def semantic_batch_arrays(run, turns, offsets, indexes, semantics):
    numeric = batch_arrays(run, turns, offsets, indexes)
    length = numeric[1].shape[1]
    local = np.zeros((len(indexes), length, EMBEDDING_DIMENSIONS), dtype=np.float32)
    for j, i in enumerate(indexes):
        selected = semantics['turn_indexes'][offsets[i]:offsets[i+1]]
        local[j,:len(selected)] = semantics['vectors'][selected]
    goals = semantics['vectors'][semantics['goal_indexes'][indexes]]
    return *numeric, torch.from_numpy(goals), torch.from_numpy(local)


def semantic_predict(network, run, turns, offsets, indexes, semantics, batch_size=64):
    network.eval()
    device = next(network.parameters()).device
    probabilities, successes, candidates = [], [], []
    with torch.inference_mode():
        for start in range(0, len(indexes), batch_size):
            selected = indexes[start:start+batch_size]
            batch = tuple(value.to(device) for value in semantic_batch_arrays(run, turns, offsets, selected, semantics))
            labels, ps, onset = network(*batch)
            probabilities.append(labels.softmax(-1).cpu().numpy())
            successes.append(ps.sigmoid().cpu().numpy())
            onset = onset.softmax(1).cpu().numpy()
            for j, i in enumerate(selected):
                n = int(offsets[i+1]-offsets[i]); scores = onset[j,:n]
                sums = np.array([scores[max(0,t-2):min(n,t+3)].sum(0) for t in range(n)])
                candidates.append([-1, *sums.argmax(0).tolist()])
    return np.concatenate(probabilities), np.concatenate(successes), np.array(candidates)
