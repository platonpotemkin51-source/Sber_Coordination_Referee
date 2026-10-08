"""Small multitask transformer over observed event features."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SequenceNetwork(nn.Module):
    def __init__(self, run_features, turn_features, width=96):
        super().__init__()
        self.run_encoder=nn.Sequential(nn.Linear(run_features,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(.1))
        self.turn_encoder=nn.Sequential(nn.Linear(turn_features,width),nn.LayerNorm(width),nn.GELU())
        layer=nn.TransformerEncoderLayer(width,4,dim_feedforward=width*3,dropout=.1,batch_first=True,norm_first=True,activation='gelu')
        self.sequence=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.classifier=nn.Sequential(nn.Linear(width*3,width),nn.GELU(),nn.Dropout(.1),nn.Linear(width,7))
        self.success=nn.Sequential(nn.Linear(width*3,width),nn.GELU(),nn.Linear(width,1))
        self.onset=nn.Linear(width,6)

    def forward(self, run, turns, padding):
        global_token=self.run_encoder(run)
        tokens=self.turn_encoder(turns)
        position=torch.arange(tokens.shape[1],device=tokens.device,dtype=tokens.dtype)[None,:,None]
        dimensions=torch.arange(tokens.shape[2],device=tokens.device,dtype=tokens.dtype)[None,None,:]
        positional=torch.sin(position/torch.pow(10000,2*torch.floor(dimensions/2)/tokens.shape[2]) + (dimensions%2)*torch.pi/2)
        tokens=tokens+.1*positional
        joined=torch.cat([global_token[:,None],tokens],dim=1)
        full_padding=torch.cat([torch.zeros((len(run),1),device=run.device,dtype=torch.bool),padding],dim=1)
        encoded=self.sequence(joined,src_key_padding_mask=full_padding)
        local=encoded[:,1:]
        valid=(~padding)[:,:,None]
        mean=(local*valid).sum(1)/valid.sum(1).clamp(min=1)
        maximum=local.masked_fill(padding[:,:,None],-1e4).amax(1)
        fused=torch.cat([encoded[:,0],mean,maximum],dim=1)
        return self.classifier(fused),self.success(fused)[:,0],self.onset(local).masked_fill(padding[:,:,None],-1e4)


def signed_log(array):
    return np.sign(array)*np.log1p(np.abs(array))


def normalize_training(data, training):
    run=signed_log(data['X'].to_numpy(dtype=np.float32))
    turns=signed_log(data['turn_X'])
    run_mean=run[training].mean(0); run_std=np.maximum(run[training].std(0),.05)
    # Accumulate moments without allocating another full training-turn matrix.
    total=np.zeros(turns.shape[1],dtype=np.float64); squares=total.copy(); count=0
    for i in training:
        block=turns[data['offsets'][i]:data['offsets'][i+1]]
        total+=block.sum(0,dtype=np.float64)
        squares+=(block.astype(np.float64)**2).sum(0)
        count+=len(block)
    turn_mean=(total/count).astype(np.float32)
    turn_std=np.maximum(np.sqrt(np.maximum(squares/count-(total/count)**2,0)),.05).astype(np.float32)
    run=np.clip((run-run_mean)/run_std,-8,8)
    turns=np.clip((turns-turn_mean)/turn_std,-8,8)
    return run,turns,{'run_mean':run_mean,'run_std':run_std,'turn_mean':turn_mean,'turn_std':turn_std}


def batch_arrays(run, turns, offsets, indexes):
    length=max(int(offsets[i+1]-offsets[i]) for i in indexes)
    local=np.zeros((len(indexes),length,turns.shape[1]),dtype=np.float32)
    padding=np.ones((len(indexes),length),dtype=bool)
    for j,i in enumerate(indexes):
        block=turns[offsets[i]:offsets[i+1]]
        local[j,:len(block)]=block
        padding[j,:len(block)]=False
    return torch.from_numpy(run[indexes]),torch.from_numpy(local),torch.from_numpy(padding)


def predict(network, run, turns, offsets, indexes, batch_size=64):
    network.eval()
    device=next(network.parameters()).device
    probabilities,success,candidates=[],[],[]
    with torch.inference_mode():
        for start in range(0,len(indexes),batch_size):
            selected=indexes[start:start+batch_size]
            batch=tuple(x.to(device) for x in batch_arrays(run,turns,offsets,selected))
            label,ps,onset=network(*batch)
            probabilities.append(label.softmax(-1).cpu().numpy())
            success.append(ps.sigmoid().cpu().numpy())
            onset=onset.softmax(1).cpu().numpy()
            for j,i in enumerate(selected):
                n=int(offsets[i+1]-offsets[i])
                scores=onset[j,:n]
                sums=np.array([scores[max(0,t-2):min(n,t+3)].sum(0) for t in range(n)])
                candidates.append([-1,*sums.argmax(0).tolist()])
    return np.concatenate(probabilities),np.concatenate(success),np.array(candidates)
