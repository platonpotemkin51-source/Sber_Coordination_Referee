"""Optional local encoder features, with label-free dimensional projection."""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from referee.data import CODE_HASH
SEMANTIC_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def semantic_features(data, encoder="weights/encoder", cache="outputs/cache", dimensions=32):
    encoder=Path(encoder)
    weights=encoder/"model.safetensors"
    if not weights.exists():
        weights=encoder/"pytorch_model.bin"
    model_sha=hashlib.sha256(weights.read_bytes()).hexdigest()[:12]
    target=Path(cache)/f"semantic-{data['source_sha256'][:16]}-{model_sha}-{dimensions}-{CODE_HASH}-{SEMANTIC_HASH}.joblib"
    if target.exists():
        values=joblib.load(target)
    else:
        import torch
        from transformers import AutoModel,AutoTokenizer
        torch.set_num_threads(4)
        tokenizer=AutoTokenizer.from_pretrained(str(encoder),local_files_only=True)
        model=AutoModel.from_pretrained(str(encoder),local_files_only=True).eval()
        run_texts=[]
        for document in data["documents"]:
            lines=document.splitlines()
            goal=lines[0].removeprefix("GOAL ").replace("_"," ")
            messages=[line.split(" ",1)[-1].replace("_"," ") for line in lines[1:]]
            run_texts.append((goal,messages))
        corpus=sorted({s for goal,msgs in run_texts for s in [goal]+msgs})
        print(f"Encoding {len(corpus)} unique texts",flush=True)
        vectors=[]
        for start in range(0,len(corpus),64):
            inputs=tokenizer(corpus[start:start+64],padding=True,truncation=True,max_length=96,return_tensors="pt")
            with torch.inference_mode():
                hidden=model(**inputs).last_hidden_state
                mask=inputs["attention_mask"].unsqueeze(-1)
                embeddings=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                embeddings=torch.nn.functional.normalize(embeddings,p=2,dim=1)
                vectors.append(embeddings.cpu().numpy())
        embedding=np.concatenate(vectors)
        lookup={text:i for i,text in enumerate(corpus)}
        projection=np.random.default_rng(42).normal(size=(embedding.shape[1],dimensions)).astype(np.float32)/np.sqrt(dimensions)
        projected=embedding@projection
        run_rows,turn_rows=[],[]
        for goal,messages in run_texts:
            g=embedding[lookup[goal]]
            indexes=[lookup[x] for x in messages]
            full=embedding[indexes]
            projected_messages=projected[indexes]
            cosine=full@g
            row={f"semantic_goal_{d}":projected[lookup[goal],d] for d in range(dimensions)}
            for d in range(dimensions):
                row[f"semantic_mean_{d}"]=float(projected_messages[:,d].mean())
                row[f"semantic_std_{d}"]=float(projected_messages[:,d].std())
                row[f"semantic_tail_{d}"]=float(projected_messages[-max(1,len(messages)//3):,d].mean())
            for q in (0,.25,.5,.75,1):
                row[f"semantic_goal_q{q}"]=float(np.quantile(cosine,q))
            row["semantic_goal_head"]=float(cosine[:max(1,len(messages)//3)].mean())
            row["semantic_goal_tail"]=float(cosine[-max(1,len(messages)//3):].mean())
            row["semantic_goal_delta"]=row["semantic_goal_tail"]-row["semantic_goal_head"]
            row["semantic_goal_low_share"]=float((cosine<.2).mean())
            local=[]
            for t in range(len(messages)):
                before=cosine[max(0,t-5):t]
                after=cosine[t:min(len(messages),t+6)]
                similarity=full[t]@full.T
                similarity[t]=-1
                local.append([*projected_messages[t],cosine[t],float(before.mean()) if len(before) else 0,float(after.mean()),float(similarity.max()) if len(messages)>1 else 0])
            run_rows.append(row)
            turn_rows.append(np.array(local,dtype=np.float32))
        values={"run":pd.DataFrame(run_rows).astype(np.float32),"turn":np.concatenate(turn_rows),"turn_columns":[f"semantic_message_{d}" for d in range(dimensions)]+["semantic_goal_cosine","semantic_goal_before","semantic_goal_after","semantic_repetition_cosine"],"encoder_sha":model_sha}
        target.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(values,target,compress=3)
    if len(values["turn"]) != len(data["turn_X"]):
        raise ValueError("Encoder texts and event turns do not align")
    out=dict(data)
    out["X"]=pd.concat([data["X"],values["run"]],axis=1)
    out["turn_X"]=np.hstack([data["turn_X"],values["turn"]])
    out["turn_columns"]=data["turn_columns"]+values["turn_columns"]
    out["encoder_sha"]=values["encoder_sha"]
    return out
