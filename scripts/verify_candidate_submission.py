"""Execute the delivered ZIP in isolation, including the platform's reduced input."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile

import pandas as pd

LABELS = {"clean", "dropped_handoff", "duplicated_work", "deadlock", "conflict",
          "goal_drift", "runaway_loop"}
COLUMNS = ["run_id", "label", "success", "fault_turn"]


def validate(prediction, source):
    assert prediction.columns.tolist() == COLUMNS
    assert prediction.run_id.tolist() == source.run_id.tolist(), "IDs/order changed"
    assert not prediction.isna().any().any()
    assert set(prediction.label) <= LABELS
    assert set(prediction.success) <= {0, 1}
    assert (prediction.fault_turn % 1 == 0).all()
    assert (prediction.fault_turn >= -1).all()
    assert (prediction.loc[prediction.label == "clean", "fault_turn"] == -1).all()
    if "messages" in source:
        lengths = source.messages.map(
            lambda value: len(json.loads(value) or []) if value.strip() else 0)
        assert ((prediction.fault_turn == -1) | (prediction.fault_turn < lengths)).all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    archive, test = Path(args.archive).resolve(), Path(args.test).resolve()
    work = Path(tempfile.mkdtemp(prefix="referee-verified-"))
    with ZipFile(archive) as package:
        names = package.namelist()
        assert len(names) == len(set(names))
        assert package.testzip() is None
        package.extractall(work / "package")
    (work / "execution").mkdir()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    rows = pd.read_csv(test, dtype={"run_id": str}, keep_default_na=False)
    small = rows.iloc[:3].copy()
    reduced = pd.DataFrame({"run_id": [f"smoke_{i}" for i in range(2000)]})
    empty_messages = small.copy()
    empty_messages["messages"] = "[]"
    empty_messages["topology"] = "[]"
    null_messages = empty_messages.copy()
    null_messages["messages"] = ["null", "", "[]"]
    mixed = small.copy()
    mixed.loc[mixed.index[1], "messages"] = "[]"
    mixed["topology"] = "[]"
    telemetry = small.copy()
    def damage(value):
        messages = json.loads(value)
        for message in messages:
            message.pop("intent", None)
            message.pop("refs", None)
            message["text"] = message.get("text", "")[:12]
        return json.dumps(messages)
    telemetry["messages"] = telemetry.messages.map(damage)
    cases = {"reduced_2000": reduced, "empty_file": small.iloc[:0],
             "empty_messages": empty_messages, "null_messages": null_messages,
             "small_real": small, "mixed_empty": mixed, "lost_telemetry": telemetry,
             "string_ids": pd.DataFrame({"run_id": ["0001", "NA", "null"]}),
             "full_test": rows}
    report = {"archive": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "work_directory": str(work), "python": sys.version, "cases": {}}
    report["dependencies"] = {}
    for dependency in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "torch"):
        try:
            report["dependencies"][dependency] = version(dependency)
        except PackageNotFoundError:
            report["dependencies"][dependency] = None
    predictions = {}
    for name, frame in cases.items():
        path, output, log = work / f"{name}.csv", work / f"{name}-predictions.csv", work / f"{name}.log"
        frame.to_csv(path, index=False)
        started = time.monotonic()
        with log.open("w") as stream:
            completed = subprocess.run(
                [sys.executable, "-u", str(work / "package/solution.py"), "--train",
                 str(work / "intentionally-absent-train.csv"), "--test", str(path),
                 "--output", str(output)], cwd=work / "execution", env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
        assert completed.returncode == 0, f"{name}: {log.read_text()[-5000:]}"
        prediction = pd.read_csv(output, dtype={"run_id": str}, keep_default_na=False)
        validate(prediction, frame)
        if name in {"reduced_2000", "empty_messages", "null_messages", "string_ids"}:
            assert (prediction.label == "clean").all()
            assert (prediction.success == 1).all()
        predictions[name] = prediction
        report["cases"][name] = {"rows": len(prediction), "seconds": round(time.monotonic() - started, 3),
                                  "returncode": completed.returncode, "predictions": str(output)}
        print(f"PASS {name}: {report['cases'][name]}", flush=True)
    pd.testing.assert_frame_equal(predictions["small_real"], predictions["full_test"].iloc[:3])
    assert predictions["mixed_empty"].iloc[1].label == "clean"
    report["small_vs_full_equal"] = True
    report["full_label_counts"] = predictions["full_test"].label.value_counts().to_dict()
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"Verified archive: {archive}\nReport: {args.report}", flush=True)


if __name__ == "__main__":
    main()
