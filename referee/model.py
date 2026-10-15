"""CPU ensemble and supervised turn classification."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from referee.data import LABELS


def booster(trees=650,leaves=31,seed=42):
    return LGBMClassifier(n_estimators=trees,learning_rate=.04,num_leaves=leaves,
        min_child_samples=25,colsample_bytree=.85,subsample=.9,subsample_freq=1,
        reg_lambda=2,n_jobs=8,random_state=seed,verbosity=-1)


class RunModel:
    def __init__(self, mode="struct", trees=650, leaves=31, text_weight=.25):
        self.mode,self.trees,self.leaves,self.text_weight=mode,trees,leaves,text_weight
        self.text_model=None

    def matrix(self,data,idx):
        if self.mode == "baseline":
            return data["X"].iloc[idx][self.columns]
        columns=getattr(self,"training_columns",list(data["X"].columns))
        return data["X"].iloc[idx].reindex(columns=columns,fill_value=0)

    def fit(self,data,idx):
        from baseline.features import extract_features
        # Capture exactly the starter feature names instead of a fragile prefix list.
        self.columns=list(extract_features({"run_id":"", "goal":"", "agents":[], "messages":[], "artifacts":[], "shared_state":[], "topology":{}}))
        self.training_columns=list(data["X"].columns)
        self.clf=booster(self.trees,self.leaves).fit(self.matrix(data,idx),data["y"][idx])
        self.suc=booster(self.trees,self.leaves).fit(self.matrix(data,idx),data["success"][idx])
        if self.mode in ("text","ensemble"):
            self.vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_df=.98,max_features=65000,sublinear_tf=True,dtype=np.float32)
            xt=self.vectorizer.fit_transform([data["documents"][i] for i in idx])
            self.text_model=LogisticRegression(C=3,max_iter=500).fit(xt,data["y"][idx])
        return self

    def predict(self,data,idx):
        probabilities=self.clf.predict_proba(self.matrix(data,idx))
        if self.text_model is not None:
            xt=self.vectorizer.transform([data["documents"][i] for i in idx])
            ptext=self.text_model.predict_proba(xt)
            probabilities=(1-self.text_weight)*probabilities+self.text_weight*ptext
        return probabilities,self.suc.predict_proba(self.matrix(data,idx))[:,1]


class TurnModel:
    """One multiclass classifier: background plus six fault classes per turn."""
    def fit(self,data,idx):
        self.turn_columns=list(data["turn_columns"])
        selections,targets,weights=[],[],[]
        for i in idx:
            start,end=data["offsets"][i:i+2]
            selections.extend(range(start,end))
            y=np.zeros(end-start,dtype=np.int8)
            if data["y"][i] != 0:
                # Annotate exact onset; nearby background stays background.
                y[int(data["fault"][i])]=data["y"][i]
            targets.extend(y)
            weights.extend(np.where(y>0,15.,1.) / max(1,end-start)**.3)
        categorical=[j for j,c in enumerate(self.turn_columns) if c.startswith("template_id_")]
        self.clf=booster(trees=550,leaves=31).fit(data["turn_X"][selections],np.array(targets),sample_weight=np.array(weights),categorical_feature=categorical if categorical else "auto")
        return self

    def predict(self,data,idx,window=2):
        blocks=[np.arange(*data["offsets"][i:i+2]) for i in idx if data["offsets"][i+1] > data["offsets"][i]]
        selections=np.concatenate(blocks) if blocks else np.empty(0,dtype=int)
        columns=getattr(self,"turn_columns",data["turn_columns"])
        feature_indexes=[data["turn_columns"].index(c) for c in columns]
        p=(self.clf.predict_proba(data["turn_X"][selections][:,feature_indexes])
           if len(selections) else np.empty((0,len(self.clf.classes_))))
        turns=np.full((len(idx),len(LABELS)),-1,dtype=int)
        evidence=np.zeros((len(idx),len(LABELS)))
        offset=0
        for j,i in enumerate(idx):
            n=data["offsets"][i+1]-data["offsets"][i]
            local=p[offset:offset+n]
            offset+=n
            if n == 0:
                evidence[j,0]=1.
                continue
            for c in range(1,len(LABELS)):
                probs=local[:,c]
                evidence[j,c]=probs.max(initial=0)
                # Smooth with the evaluation tolerance; avoid np.convolve shape
                # surprises for logs shorter than the window.
                sums=np.array([probs[max(0,t-window):min(n,t+window+1)].sum() for t in range(n)])
                turns[j,c]=int(sums.argmax()) if n else -1
            evidence[j,0]=1-evidence[j,1:].max(initial=0)
        return turns,evidence
