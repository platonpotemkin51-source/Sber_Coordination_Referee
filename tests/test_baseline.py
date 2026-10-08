"""Sanity checks for the starter kit: features, localiser, submission format."""

import json
import math
import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "data", "train.csv")

from features import extract_features, parse_run   # noqa: E402
from localize import localize                      # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(TRAIN),
    reason="run `bash scripts/prepare_data.sh` first")


@pytest.fixture(scope="module")
def runs():
    df = pd.read_csv(TRAIN, nrows=200)
    return df, [parse_run(r) for r in df.to_dict("records")]


def test_runs_parse(runs):
    _, parsed = runs
    for run in parsed:
        assert run["messages"] and isinstance(run["messages"], list)
        assert isinstance(run["topology"], dict)
        assert [m["t"] for m in run["messages"]] == list(range(len(run["messages"])))


def test_features_are_finite_and_stable(runs):
    _, parsed = runs
    keys = None
    for run in parsed:
        f = extract_features(run)
        if keys is None:
            keys = set(f)
        assert set(f) == keys
        for k, v in f.items():
            assert math.isfinite(float(v)), k


def test_localizer_stays_inside_the_run(runs):
    df, parsed = runs
    for run, label in zip(parsed, df["label"]):
        t = localize(run, label)
        assert -1 <= t < len(run["messages"])
        if label == "clean":
            assert t == -1


def test_labels_and_fault_turn_are_consistent(runs):
    df, _ = runs
    clean = df[df.label == "clean"]
    faulty = df[df.label != "clean"]
    assert (clean.fault_turn == -1).all()
    assert (faulty.fault_turn >= 0).all()
    assert set(df["success"].unique()) <= {0, 1}


def test_submission_validator_accepts_sample():
    sample = os.path.join(ROOT, "data", "sample_submission.csv")
    test = os.path.join(ROOT, "data", "test.csv")
    if not os.path.exists(test):
        pytest.skip("run `bash scripts/prepare_data.sh` first")
    subprocess.check_call(
        [sys.executable, os.path.join(ROOT, "scripts", "validate_submission.py"),
         "--pred", sample, "--test", test], stdout=subprocess.DEVNULL)
