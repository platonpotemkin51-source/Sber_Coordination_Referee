"""Regression coverage for the actual candidate entry point packed into the ZIP."""
import builtins
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


spec = importlib.util.spec_from_file_location(
    "candidate_solution", Path(__file__).resolve().parents[1] / "scripts/candidate_solution.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


@pytest.mark.parametrize("kind", ["reduced", "empty", "no_messages", "null_messages"])
def test_fallback_does_not_load_models(kind, monkeypatch):
    frame = pd.DataFrame({"run_id": ["0001", "NA", "smoke_3"]})
    if kind != "reduced":
        for column in candidate.RUN_COLUMNS:
            frame[column] = "[]"
        frame["messages"] = {"empty": "[]", "no_messages": "[]", "null_messages": "null"}[kind]
    if kind == "empty":
        frame = frame.iloc[:0]
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"joblib", "torch", "lightgbm", "referee", "v15_sequence_inference"}:
            raise AssertionError(f"Fallback imported model dependency: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = candidate.predict_frame(frame, "missing.csv", "cpu")
    assert result.columns.tolist() == candidate.OUTPUT_COLUMNS
    assert result.run_id.tolist() == frame.run_id.tolist()
    assert result.label.tolist() == ["clean"] * len(frame)
    assert result.success.tolist() == [1] * len(frame)
    assert result.fault_turn.tolist() == [-1] * len(frame)


@pytest.mark.parametrize("value", ["[]", "null", "", " ", np.nan])
def test_no_messages(value):
    assert candidate.has_messages(value) is False


def test_active_message():
    assert candidate.has_messages('[{"t":0,"text":"hello"}]') is True


def test_invalid_messages_are_not_silently_hidden():
    with pytest.raises(ValueError):
        candidate.has_messages('{"wrong":"schema"}')


@pytest.mark.parametrize("indexes", [np.array([0, 1]), np.array([], dtype=int)])
def test_turn_model_with_no_active_turns(indexes):
    from referee.model import TurnModel

    class Classifier:
        classes_ = np.arange(7)

        def predict_proba(self, matrix):
            raise AssertionError("The classifier cannot predict an empty matrix")

    model = TurnModel()
    model.clf = Classifier()
    data = {"offsets": np.array([0, 0, 0]), "turn_columns": ["feature"],
            "turn_X": np.empty((0, 1))}
    turns, evidence = model.predict(data, indexes)
    assert turns.shape == evidence.shape == (len(indexes), 7)
    assert (turns == -1).all()
    assert (evidence[:, 0] == 1).all()
