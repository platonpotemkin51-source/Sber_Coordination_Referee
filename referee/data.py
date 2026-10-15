"""Feature caching and fixed splits at the run level."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from baseline.features import parse_run
from referee.events import prepare, normalize

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock", "conflict", "goal_drift", "runaway_loop"]
CODE_HASH = hashlib.sha256(Path(__file__).with_name("events.py").read_bytes()).hexdigest()[:12]


def load_data(path, cache="outputs/cache"):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    target = Path(cache)/f"{digest[:16]}-{CODE_HASH}.joblib"
    if target.exists():
        return joblib.load(target)
    df = pd.read_csv(path)
    run_rows, turn_arrays, documents, groups, offsets = [], [], [], [], [0]
    turn_columns = None
    for i, record in enumerate(df.to_dict("records")):
        run = parse_run(record)
        fr, turns, doc = prepare(run)
        if turn_columns is None and turns:
            turn_columns = sorted(turns[0])
        run_rows.append(fr)
        if turns:
            turn_arrays.append(np.array([[row.get(k,0) for k in turn_columns] for row in turns], dtype=np.float32).reshape(-1,len(turn_columns)))
        else:
            turn_arrays.append(None)
        offsets.append(offsets[-1]+len(turns))
        documents.append(doc)
        agent_ids = {a["id"]: j for j,a in enumerate(run["agents"])}
        topology_type = run["topology"].get("type") if isinstance(run["topology"], dict) else None
        signature = [normalize(run["goal"]), topology_type, [(agent_ids.get(m.get("from"),-1), agent_ids.get(m.get("to"),-1), m.get("type"), m.get("intent"), normalize(m.get("text",""))) for m in run["messages"]]]
        groups.append(hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest())
        if i % 1000 == 0:
            print(f"features {i}/{len(df)}", flush=True)
    if turn_columns is None:
        raise ValueError("Cannot infer turn feature columns: the input contains no messages in any run")
    turn_arrays = [array if array is not None else np.zeros((0, len(turn_columns)), dtype=np.float32) for array in turn_arrays]
    out = {"X": pd.DataFrame(run_rows).fillna(0).astype(np.float32), "turn_X": np.concatenate(turn_arrays), "turn_columns": turn_columns,
           "offsets": np.array(offsets), "documents": documents, "groups": np.array(groups), "source_sha256":digest,
           "run_id":df.run_id.to_numpy(), "y":np.array([LABELS.index(x) for x in df.label]) if "label" in df else None,
           "success":df.success.to_numpy() if "success" in df else None, "fault":df.fault_turn.to_numpy() if "fault_turn" in df else None}
    target.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(out,target,compress=3)
    return out


def fixed_folds(data):
    folds=np.full(len(data["y"]),-1,dtype=int)
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=20261007)
    for fold,(_,valid) in enumerate(cv.split(data["X"],data["y"],data["groups"])):
        folds[valid]=fold
    return folds


def concatenate(primary, extra):
    """Append separately sourced runs; identifiers are not used as join keys."""
    if primary["turn_columns"] != extra["turn_columns"]:
        raise ValueError("Incompatible turn features")
    out={key:primary[key] for key in ("turn_columns","source_sha256")}
    out["X"]=pd.concat([primary["X"],extra["X"]],ignore_index=True).fillna(0)
    out["turn_X"]=np.concatenate([primary["turn_X"],extra["turn_X"]])
    out["offsets"]=np.concatenate([primary["offsets"],extra["offsets"][1:]+primary["offsets"][-1]])
    out["documents"]=primary["documents"]+extra["documents"]
    for key in ("groups","run_id","y","success","fault"):
        out[key]=np.concatenate([primary[key],extra[key]])
    return out


def restrict_features(data,profile="current"):
    if profile == "current":
        return data
    schema=json.loads(Path(__file__).with_name("feature_schema_v2.json").read_text())
    out=dict(data)
    out["X"]=data["X"][schema["run_columns"]]
    selection=[data["turn_columns"].index(c) for c in schema["turn_columns"]]
    out["turn_X"]=data["turn_X"][:,selection]
    out["turn_columns"]=schema["turn_columns"]
    return out
