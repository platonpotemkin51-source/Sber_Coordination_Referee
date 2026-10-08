"""Candidate ranking from honest out-of-fold detector predictions."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMRanker


def candidate_features(data, indexes, probabilities, evidence, candidates):
    global_features=data['X'].iloc[indexes].to_numpy(dtype=np.float32)
    count=len(indexes); local_width=data['turn_X'].shape[1]
    local=np.zeros((count,7,local_width*2),dtype=np.float32)
    extras=np.zeros((count,7,8),dtype=np.float32)
    for j,i in enumerate(indexes):
        start,end=data['offsets'][i:i+2]; n=end-start
        for c in range(7):
            t=int(candidates[j,c])
            if c and 0 <= t < n:
                window=data['turn_X'][start+max(0,t-2):start+min(n,t+3)]
                local[j,c,:local_width]=window.mean(0)
                local[j,c,local_width:]=window.max(0)
            other=[k for k in range(7) if k!=c]
            extras[j,c]=[t,t/max(1,n),probabilities[j,c],evidence[j,c],probabilities[j,other].max(),evidence[j,other].max(),probabilities[j,c]-probabilities[j,other].max(),evidence[j,c]-evidence[j,other].max()]
    repeated=np.repeat(global_features[:,None],7,axis=1)
    class_code=np.broadcast_to(np.eye(7,dtype=np.float32),(count,7,7))
    outputs=np.repeat(np.hstack([probabilities,evidence])[:,None],7,axis=1)
    features=np.concatenate([repeated,local,class_code,outputs,extras],axis=2)
    return features.reshape(count*7,-1)


class CandidateRanker:
    def fit(self,data,indexes,probabilities,evidence,candidates):
        self.run_columns=list(data['X'].columns)
        self.turn_columns=list(data['turn_columns'])
        x=candidate_features(data,indexes,probabilities,evidence,candidates)
        y=(np.arange(7)[None,:]==data['y'][indexes,None]).astype(int).ravel()
        frequencies=np.bincount(data['y'][indexes],minlength=7)
        row_weights=len(indexes)/(7*np.maximum(frequencies[data['y'][indexes]],1))
        self.clf=LGBMRanker(n_estimators=450,learning_rate=.04,num_leaves=31,min_child_samples=100,reg_lambda=10,colsample_bytree=.85,n_jobs=8,random_state=42,verbosity=-1,lambdarank_truncation_level=2)
        self.clf.fit(x,y,group=np.full(len(indexes),7),sample_weight=np.repeat(row_weights,7))
        return self

    def predict(self,data,indexes,probabilities,evidence,candidates):
        out=dict(data)
        out['X']=data['X'].reindex(columns=self.run_columns,fill_value=0)
        columns=[data['turn_columns'].index(c) for c in self.turn_columns]
        out['turn_X']=data['turn_X'][:,columns]
        out['turn_columns']=self.turn_columns
        scores=self.clf.predict(candidate_features(out,indexes,probabilities,evidence,candidates)).reshape(-1,7)
        scores-=scores.max(1,keepdims=True)
        p=np.exp(scores)
        return p/p.sum(1,keepdims=True)
