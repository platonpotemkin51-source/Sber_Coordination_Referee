"""Freeze the audited .92 local-control milestone without retraining models."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import torch

from referee.data import CODE_HASH
from referee.relations import RELATIONS_HASH
from referee.pair_selector import PAIR_HASH


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="weights/v5/model.joblib")
    parser.add_argument("--pair-experiment", default="outputs/pair_selector_v15")
    parser.add_argument("--sequence", default="outputs/sequence_v6/model.pt")
    parser.add_argument("--output", default="weights/pair-v15/model.joblib")
    parser.add_argument("--validation-output", default="outputs/pair-v15-fixed/validation.json")
    args = parser.parse_args()
    started = time.monotonic()
    base_path, sequence_path = ROOT / args.base, ROOT / args.sequence
    pair_folder = ROOT / args.pair_experiment
    base_metadata = read_json(base_path.with_suffix(".json"))
    development = read_json(pair_folder / "metrics.json")
    control = read_json(pair_folder / "fold0/metrics.json")
    parameters = read_json(pair_folder / "parameters.json")
    selected = joblib.load(pair_folder / "model.joblib")
    if parameters != selected["parameters"] or parameters["selector_weight"] != .5:
        raise ValueError("Saved pair-selector parameters differ from the frozen milestone")
    certificate = selected["certificate"]
    if certificate != development["certificate"]:
        raise ValueError("Pair model and reported development have different OOF provenance")
    if not certificate.get("every_training_row_oof_once") or certificate["training_rows"] != 16000:
        raise ValueError("The milestone requires the audited 16k OOF pair selector")
    if certificate["features_code_hash"] != CODE_HASH or certificate["relations_hash"] != RELATIONS_HASH or certificate["pair_hash"] != PAIR_HASH:
        raise ValueError("Pair model provenance does not match the current extractor/selector")
    if base_metadata["sources"] != certificate["sources"] or base_metadata["rows"] != 20000:
        raise ValueError("Full base model must retain the two recorded training sources and 20k rows")
    for source, expected in zip((base_metadata["configuration"]["train"], base_metadata["configuration"]["extra_train"]), base_metadata["sources"]):
        if sha256(ROOT / source) != expected:
            raise ValueError(f"Training source has changed: {source}")
    if control["score_proxy"] != 0.9205340787305163:
        raise ValueError("This freeze script targets the exact saved .9205340787305163 control result")
    if control["meta_training_rows"] != 16000 or control["base_detector_training_rows"] != 18000:
        raise ValueError("Control training sizes do not match the recorded evaluation")
    base = joblib.load(base_path)
    if base["features_code_hash"] != CODE_HASH or base["relations_hash"] != RELATIONS_HASH:
        raise ValueError("Full base weights do not match current v5 features")
    if base["turn_weight"] != .5 or base["turn_power"] != 2:
        raise ValueError("Full base model must use the original .5/2 detector fusion")
    sequence = torch.load(sequence_path, map_location="cpu", weights_only=False)
    if sequence["features_code_hash"] != CODE_HASH or sequence["relations_hash"] != RELATIONS_HASH or sequence["epoch"] != 15:
        raise ValueError("Neural success checkpoint differs from the frozen v6 epoch 15")
    sequence_metrics = read_json(sequence_path.with_name("metrics.json"))
    if sequence_metrics["configuration"]["train"] != base_metadata["configuration"]["train"] or sequence_metrics["configuration"]["extra_train"] != base_metadata["configuration"]["extra_train"]:
        raise ValueError("Neural success source paths differ from the recorded v6 recipe")
    output = ROOT / args.output
    if output.resolve() in (base_path.resolve(), (ROOT / "weights/model.joblib").resolve()):
        raise ValueError("Freeze output must not overwrite an existing base/default model")
    output.parent.mkdir(parents=True, exist_ok=True)
    sequence_target = output.parent / "sequence-success.pt"
    shutil.copy2(sequence_path, sequence_target)
    checkpoint_sha = sha256(sequence_target)
    if checkpoint_sha != sha256(sequence_path):
        raise ValueError("Copied neural checkpoint failed integrity verification")
    neural_success = {"checkpoint": sequence_target.name, "sha256": checkpoint_sha}
    configuration = {
        **base_metadata["configuration"], "output": args.output,
        "profile": "v5", "pair_selector_weight": parameters["selector_weight"],
        "success_threshold": .45,
    }
    bundle = {
        **base, "pair_selector": selected["selector"],
        "pair_selector_weight": parameters["selector_weight"], "pair_hash": PAIR_HASH,
        "neural_success": neural_success, "success_threshold": .45,
        "configuration": configuration,
    }
    joblib.dump(bundle, output, compress=3)
    frozen_at = datetime.now(timezone.utc).isoformat()
    neural_provenance = {
        "source_checkpoint": args.sequence, "checkpoint_sha256": checkpoint_sha,
        "selected_epoch": sequence["epoch"], "training_rows": 16000,
        "excluded_outer_folds": [0, 1], "sources": base_metadata["sources"],
        "features_code_hash": sequence["features_code_hash"],
        "relations_hash": sequence["relations_hash"],
        "configuration": sequence_metrics["configuration"],
        "use": "Frozen sequence_v6 success head only; class and onset heads are not used.",
    }
    metadata = {
        **base_metadata, "configuration": configuration, "frozen_at_utc": frozen_at,
        "version": "pair_selector_v15_fixed", "runtime": "gpu",
        "runtime_reason": "The GPU runtime image supplies PyTorch 2.5.1. Neural success inference supports a CPU fallback.",
        "pair_hash": PAIR_HASH, "pair_selector_weight": parameters["selector_weight"],
        "neural_success": neural_success, "neural_provenance": neural_provenance,
        "pair_provenance": certificate,
        "base_provenance": {"path": args.base, "sha256": sha256(base_path), "training": base_metadata},
        "weights_sha256": sha256(output), "freeze_elapsed_seconds": time.monotonic() - started,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    note = (
        "The .9205340787305163 headline is a repeatedly evaluated local control fold 0, "
        "not a fresh independent holdout or platform score. Development fold 1 scored "
        ".8961458397830798 and its check-half .8878442473091998. RobustnessF1 is an "
        "approximate observable slice. Control used an 18k base detector with 16k "
        "OOF pair selector/neural success; submitted base weights use all 20k rows, "
        "with the selector and success checkpoint frozen. The user requested stopping "
        "at local score >= .92 and uploading for platform verification."
    )
    validation = {
        **control, "score_proxy": control["score_proxy"], "note": note,
        "version": "pair_selector_v15_fixed", "frozen_at_utc": frozen_at,
        "validation_splits": {
            "development_fold1": development,
            "check_half_fold1": development["check"],
            "repeated_control_fold0": control,
        },
        "submission_training": metadata,
        "pause_reason": "Reached the user's >= .92 local milestone; new model experiments stopped, package verification remains.",
        "features_code_hash": CODE_HASH, "relations_hash": RELATIONS_HASH,
        "pair_hash": PAIR_HASH, "weights_sha256": metadata["weights_sha256"],
        "neural_checkpoint_sha256": checkpoint_sha,
        "source_sha256": {
            name: sha256(ROOT / name) for name in (
                "referee/data.py", "referee/events.py", "referee/model.py",
                "referee/relations.py", "referee/pair_selector.py", "referee/sequence.py",
                "referee/neural_success.py", "solution.py", "scripts/package_submission.py",
                "scripts/pair_selector_experiment.py", "scripts/freeze_pair_solution.py",
            )
        },
    }
    validation_output = ROOT / args.validation_output
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    validation_output.write_text(json.dumps(validation, indent=2), encoding="utf8")
    print(json.dumps({"weights": str(output), "metadata": str(output.with_suffix(".json")),
                      "sequence": str(sequence_target), "validation": str(validation_output),
                      "score_proxy": control["score_proxy"], "runtime": "gpu",
                      "weights_sha256": metadata["weights_sha256"],
                      "neural_checkpoint_sha256": checkpoint_sha}, indent=2), flush=True)


if __name__ == "__main__":
    main()
