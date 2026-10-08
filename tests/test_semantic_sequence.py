"""Semantic cache integrity, exact numeric warm start, and padding semantics."""
import copy

import numpy as np
import pytest
import torch

from referee.sequence import SequenceNetwork
from referee.semantic_sequence import (SemanticSequenceNetwork, semantic_sequences,
                                        semantic_batch_arrays, text_index)


def test_semantic_cache_ignores_labels_and_rejects_changed_observable_input(tmp_path):
    encoder=tmp_path/'encoder';encoder.mkdir();(encoder/'model.safetensors').write_bytes(b'fake local test weights')
    data={'documents':['GOAL verify_budget\nhandoff verify_budget assigned\ninform result ready',
                       'GOAL verify_budget\ninform result ready'],
          'offsets':np.array([0,2,3]),'source_sha256':'test-source-123',
          'y':np.array([1,2]),'success':np.array([0,1]),'fault':np.array([0,0]),'run_id':['id1','id2']}
    calls=[]
    def fake_encode(corpus, encoder, **kwargs):
        calls.append(corpus)
        return np.random.default_rng(42).normal(size=(len(corpus),384)).astype(np.float32)
    first=semantic_sequences(data,encoder,tmp_path/'cache',encode=fake_encode)
    changed=copy.deepcopy(data);changed.update(y=np.array([6,6]),success=np.array([1,0]),fault=np.array([99,99]),run_id=['other1','other2'])
    other=semantic_sequences(changed,encoder,tmp_path/'cache',encode=fake_encode)
    assert len(calls)==1
    for key in ('vectors','goal_indexes','turn_indexes'):
        np.testing.assert_array_equal(first[key],other[key])
    np.testing.assert_allclose(np.linalg.norm(first['vectors'],axis=1),1,atol=1e-6)
    assert first['vectors'].shape==(3,384)
    assert first['turn_indexes'][1]==first['turn_indexes'][2]
    changed['documents'][1]='GOAL verify_budget\ninform changed observable text'
    with pytest.raises(ValueError,match='provenance'):
        semantic_sequences(changed,encoder,tmp_path/'cache',encode=fake_encode)
    with pytest.raises(ValueError,match='offsets'):
        text_index(data['documents'],np.array([0,1,3]))


def test_zero_semantic_residual_matches_v6_and_receives_gradient():
    torch.manual_seed(42)
    original=SequenceNetwork(2,3,24).eval()
    extended=SemanticSequenceNetwork(2,3,24).eval();extended.warm_start(original.state_dict())
    run=torch.randn(2,2);turns=torch.randn(2,4,3)
    padding=torch.tensor([[False,False,True,True],[False,False,False,False]])
    goals=torch.randn(2,384);messages=torch.randn(2,4,384)
    expected=original(run,turns,padding);actual=extended(run,turns,padding,goals,messages)
    for before,after in zip(expected,actual):
        torch.testing.assert_close(before,after,atol=0,rtol=0)
    loss=actual[0].square().mean()+actual[1].square().mean()
    loss.backward()
    assert extended.semantic_turn[-1].weight.grad.abs().sum()>0
    assert extended.semantic_goal[-1].weight.grad.abs().sum()>0


def test_semantic_padding_does_not_affect_valid_outputs_or_expand_dataset():
    rng=np.random.default_rng(12);torch.manual_seed(12)
    network=SemanticSequenceNetwork(2,3,24).eval()
    # Enable semantic information to verify masking beyond zero-init behavior.
    torch.nn.init.normal_(network.semantic_turn[-1].weight,std=.1)
    torch.nn.init.normal_(network.semantic_goal[-1].weight,std=.1)
    run=np.zeros((2,2),np.float32);turns=np.zeros((6,3),np.float32);offsets=np.array([0,2,6])
    vectors=rng.normal(size=(3,384)).astype(np.float32)
    semantics={'vectors':vectors,'goal_indexes':np.array([0,1],np.int32),'turn_indexes':np.array([1,2,0,1,2,0],np.int32)}
    batch=semantic_batch_arrays(run,turns,offsets,np.array([0,1]),semantics)
    assert semantics['vectors'].shape==(3,384)
    assert semantics['turn_indexes'].shape==(6,)
    np.testing.assert_array_equal(batch[-1][0,:2].numpy(),vectors[[1,2]])
    modified=list(batch);modified[-1]=batch[-1].clone();modified[-1][0,2:]=999
    with torch.inference_mode():
        expected=network(*batch);actual=network(*modified)
    for before,after in zip(expected,actual):
        torch.testing.assert_close(before,after,atol=1e-6,rtol=1e-6)
    assert torch.all(actual[-1][0,2:]==-1e4)
