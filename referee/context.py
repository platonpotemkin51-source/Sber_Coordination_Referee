"""Condition turn classification on training-selected global run features."""
from __future__ import annotations

import numpy as np


def selected_context_columns(run_model, label_count=50, success_count=30):
    columns=np.array(run_model.training_columns)
    selected=set(columns[np.argsort(run_model.clf.feature_importances_)[-label_count:]])
    selected.update(columns[np.argsort(run_model.suc.feature_importances_)[-success_count:]])
    return sorted(selected)


def global_context_features(data, columns):
    features=data['X'].reindex(columns=columns,fill_value=0).to_numpy(dtype=np.float32)
    owner=np.repeat(np.arange(len(features)),np.diff(data['offsets']))
    out=dict(data)
    out['turn_X']=np.hstack([data['turn_X'],features[owner]])
    out['turn_columns']=data['turn_columns']+['context_'+c for c in columns]
    return out
