"""Train a small train-only text CNN over ordered run messages."""
from __future__ import annotations

import json
import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
HANDOFF=ROOT/"handoff/v15/assembled"
sys.path.insert(0,str(HANDOFF)); sys.path.insert(0,str(ROOT/"scripts"))
from referee.data import concatenate,load_data
from referee.relations import relation_features
import v15_mix_audit as audit

EXPORT=HANDOFF/"evaluation_export"
OUT=HANDOFF/"outputs"
DEST=ROOT/"outputs/token-cnn-v15"
TOKEN_RE=re.compile(r"[a-z0-9_]+")
MAX_VOCAB=50000
SEQ_LEN=512
TEXT_GATE=np.asarray([.1,.1,.5,.3,0.,0.,.2],dtype=np.float32)

class TextCNN(nn.Module):
    def __init__(self,vocab_size,embedding=64,channels=96,dropout=.35):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding,padding_idx=0)
        self.conv3=nn.Conv1d(embedding,channels,3,padding=1)
        self.conv5=nn.Conv1d(embedding,channels,5,padding=2)
        self.dropout=nn.Dropout(dropout)
        self.head=nn.Sequential(nn.Linear(channels*4,192),nn.GELU(),nn.Dropout(dropout),nn.Linear(192,7))
    def forward(self,ids):
        mask=ids.ne(0)
        x=self.embedding(ids).transpose(1,2)
        features=[]
        for conv in (self.conv3,self.conv5):
            z=torch.nn.functional.gelu(conv(x))
            z=z.masked_fill(~mask[:,None,:],-1e4)
            maximum=z.amax(2)
            values=torch.where(mask[:,None,:],z,torch.zeros_like(z))
            mean=values.sum(2)/mask.sum(1,keepdim=True).clamp(min=1)
            features.extend((maximum,mean))
        return self.head(self.dropout(torch.cat(features,dim=1)))

def tokenize(text): return TOKEN_RE.findall(text.lower())

def encode(texts,vocab):
    result=np.zeros((len(texts),SEQ_LEN),dtype=np.int32)
    for i,text in enumerate(texts):
        tokens=tokenize(text)
        if len(tokens)>SEQ_LEN:
            half=SEQ_LEN//2; tokens=tokens[:half]+tokens[-(SEQ_LEN-half):]
        result[i,:len(tokens)]=[vocab.get(token,1) for token in tokens]
    return result

@torch.inference_mode()
def predict(model,features,device,batch=128):
    model.eval(); result=[]
    for start in range(0,len(features),batch):
        x=torch.from_numpy(features[start:start+batch].astype(np.int64)).to(device)
        result.append(torch.softmax(model(x),dim=1).cpu().numpy())
    return np.concatenate(result)

def norm(q):
    q=np.maximum(np.asarray(q,dtype=np.float64),1e-12)
    return q/q.sum(1,keepdims=True)

def blend(base,text,gates): return norm((1-gates[None,:])*base+gates[None,:]*text)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--batch-size",type=int,default=64)
    args=parser.parse_args()
    started=time.monotonic(); torch.set_num_threads(4); torch.manual_seed(20261013)
    cache="/private/tmp/v15_oof_feature_cache"
    paths=[HANDOFF/"data/competition/train.csv",HANDOFF/"data/train.csv"]
    pieces=[relation_features(load_data(path,cache=cache),path,cache=cache) for path in paths]
    data=concatenate(*pieces)
    population=pd.read_csv(EXPORT/"population_and_truth.csv").set_index("global_index")
    train=population.index[population.meta_and_neural_training.to_numpy(bool)].to_numpy(dtype=int)
    dev=audit.load_slice("development_fold1",None,OUT/"pair_selector_v15/validation.npz")
    fold0=audit.load_slice("fold0",None,OUT/"pair_selector_v15/fold0/validation.npz")
    dev_ids=dev["membership"].official_row_index.to_numpy(); fold_ids=fold0["membership"].official_row_index.to_numpy()
    check_truth=pd.read_csv(EXPORT/"check_half_truth.csv"); check_membership=pd.read_csv(EXPORT/"check_half_membership.csv")
    check_ids=check_membership.official_row_index.to_numpy(); position={int(v):i for i,v in enumerate(dev_ids)}
    check_pos=np.asarray([position[int(v)] for v in check_ids]); selection_pos=np.asarray([i for i,v in enumerate(dev_ids) if int(v) not in set(check_ids)])
    selection={k:(v.iloc[selection_pos].reset_index(drop=True) if k in ("truth","membership") else v[selection_pos]) for k,v in dev.items()}
    check={"truth":check_truth,"membership":check_membership,**{k:v[check_pos] for k,v in dev.items() if k not in ("truth","membership")}}
    base=np.load(ROOT/"outputs/char-tfidf-v15-experiment/predictions.npz")
    baseline_dev,baseline_0=base["baseline_dev"],base["baseline_fold0"]

    tokenized=[tokenize(data["documents"][i]) for i in train]
    counts=Counter(token for row in tokenized for token in row)
    vocab={"<PAD>":0,"<UNK>":1}
    vocab.update({token:i+2 for i,(token,count) in enumerate((x for x in counts.most_common(MAX_VOCAB-2) if x[1]>=2))})
    train_x=encode([data["documents"][i] for i in train],vocab)
    dev_x=encode([data["documents"][i] for i in dev_ids],vocab)
    fold_x=encode([data["documents"][i] for i in fold_ids],vocab)
    y=data["y"][train].astype(np.int64)
    device=torch.device("cpu")
    model=TextCNN(len(vocab)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    rng=np.random.default_rng(20261013)
    best=-1; best_state=None; best_epoch=0; history=[]
    for epoch in range(1,args.epochs+1):
        tick=time.monotonic(); model.train(); order=rng.permutation(len(train)); losses=[]
        for start in range(0,len(order),args.batch_size):
            ix=order[start:start+args.batch_size]
            xb=torch.from_numpy(train_x[ix].astype(np.int64)); yb=torch.from_numpy(y[ix])
            logits=model(xb); loss=nn.functional.cross_entropy(logits,yb,label_smoothing=.02)
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.); optimizer.step()
            losses.append(float(loss.detach()))
        pdev=predict(model,dev_x,device)
        candidate=blend(baseline_dev,pdev,TEXT_GATE)
        score=audit.score_probability(selection,candidate[selection_pos])["score_proxy"]
        record={"epoch":epoch,"loss":float(np.mean(losses)),"seconds":round(time.monotonic()-tick,2),"selection_score_proxy_fixed_gate":score}
        history.append(record); print(json.dumps(record),flush=True)
        if score>best:
            best=score; best_epoch=epoch; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pdev=predict(model,dev_x,device); p0=predict(model,fold_x,device)
    # Keep the fixed gate as the primary selection; run a small final gate grid on selection only.
    gates=TEXT_GATE.copy()
    for _ in range(2):
        for cls in range(7):
            trials=[]
            for value in (0.,.1,.2,.3,.4,.5,.75,1.):
                trial=gates.copy(); trial[cls]=value
                q=blend(baseline_dev,pdev,trial)
                trials.append((audit.score_probability(selection,q[selection_pos])["score_proxy"],value))
            gates[cls]=max(trials)[1]
    qd=blend(baseline_dev,pdev,gates); q0=blend(baseline_0,p0,gates)
    result={"training_rows":len(train),"vocabulary_size":len(vocab),"sequence_length":SEQ_LEN,
        "selected_epoch":best_epoch,"selection_fixed_gate_score":best,
        "fixed_gate":{"selection":audit.score_probability(selection,blend(baseline_dev,pdev,TEXT_GATE)[selection_pos]),
                      "check":audit.score_probability(check,blend(baseline_dev,pdev,TEXT_GATE)[check_pos]),
                      "fold0":audit.score_probability(fold0,blend(baseline_0,p0,TEXT_GATE))},
        "class_gates":gates.tolist(),
        "classwise":{"selection":audit.score_probability(selection,qd[selection_pos]),
                      "development":audit.score_probability(dev,qd),
                      "check":audit.score_probability(check,qd[check_pos]),
                      "fold0":audit.score_probability(fold0,q0)},
        "history":history,"elapsed_seconds":time.monotonic()-started}
    DEST.mkdir(parents=True,exist_ok=True)
    torch.save({"state":best_state,"vocab":vocab,"sequence_length":SEQ_LEN,"epoch":best_epoch},DEST/"model.pt")
    (DEST/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savez_compressed(DEST/"predictions.npz",dev=pdev,fold0=p0,dev_blend=qd,fold0_blend=q0,gates=gates)
    print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2),flush=True)

if __name__=="__main__": main()
