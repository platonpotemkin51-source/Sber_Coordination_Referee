"""Build a compact, explicit offline submission; no datasets or caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="submission-v5-fixed.zip")
    ap.add_argument("--weights",default="weights/model.joblib")
    ap.add_argument("--validation-metrics",default="outputs/v5-fixed/validation.json")
    args=ap.parse_args()
    weight_path=Path(args.weights)
    metadata_path=weight_path.with_suffix(".json")
    metadata=json.loads((ROOT/metadata_path).read_text())
    sources=["solution.py","requirements.txt","referee/__init__.py","referee/data.py","referee/events.py","referee/model.py","referee/feature_schema_v2.json","baseline/features.py","baseline/localize.py","scripts/fit_solution.py"]
    if metadata.get("relations_hash"):
        sources.append("referee/relations.py")
        relation_hash=hashlib.sha256((ROOT/"referee/relations.py").read_bytes()).hexdigest()[:12]
        if relation_hash != metadata["relations_hash"]:
            raise RuntimeError("Weights do not match the current relation code")
    if metadata.get("pair_hash"):
        sources.append("referee/pair_selector.py")
        pair_hash=hashlib.sha256((ROOT/"referee/pair_selector.py").read_bytes()).hexdigest()[:12]
        if pair_hash != metadata["pair_hash"]:
            raise RuntimeError("Weights do not match the current pair selector code")
    neural=metadata.get("neural_success")
    extra_weights=[]
    if neural:
        sources.extend(["referee/sequence.py","referee/neural_success.py"])
        checkpoint_name=Path(neural["checkpoint"])
        if checkpoint_name.name != str(checkpoint_name):
            raise ValueError("Neural success checkpoint must be in the selected weights directory")
        checkpoint=ROOT/weight_path.parent/checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if neural.get("sha256"):
            actual_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if actual_sha != neural["sha256"]:
                raise RuntimeError("Neural success checkpoint SHA256 does not match metadata")
        extra_weights.append((checkpoint,"weights/sequence-success.pt"))
    for source in sources:
        if not (ROOT/source).is_file():
            raise FileNotFoundError(source)
    code_hash=hashlib.sha256((ROOT/"referee/events.py").read_bytes()).hexdigest()[:12]
    if metadata["features_code_hash"] != code_hash:
        raise RuntimeError("Weights do not match the current feature code")
    if not (ROOT/weight_path).is_file():
        raise FileNotFoundError(ROOT/weight_path)
    validation=json.loads((ROOT/args.validation_metrics).read_text())
    profile=metadata["configuration"].get("profile","v2")
    validation_note=validation.get("note", "This is not an official leaderboard measurement.")
    if neural:
        runtime_note="Use the GPU runtime image, which provides PyTorch. Inference can use CUDA or CPU inside that image. Dependencies are already present on the platform; no installation or network is required."
    else:
        runtime_note="Use the CPU runtime image. Dependencies are already present on the platform; no installation, GPU or network is required."
    if metadata.get("pair_hash"):
        training_note="Weights trained offline on official and demonstration train files. Tree, cross-fitted pair selector and frozen neural success checkpoint provenance and parameters: weights/model.json."
        control_note="The reported result is a local score proxy from the previously opened control fold, not a fresh independent holdout or an official platform score."
    else:
        training_note=f"Weights trained offline on official and demonstration train files. Training provenance and parameters: weights/model.json. To retrain: python scripts/fit_solution.py --train <official_train.csv> --extra-train <demo_train.csv> --profile {profile} --success-threshold .45."
        control_note=""
    target=ROOT/args.output
    with ZipFile(target,"w",compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for source in sources:
            archive.write(ROOT/source,source)
        archive.write(ROOT/weight_path,"weights/model.joblib")
        archive.write(ROOT/metadata_path,"weights/model.json")
        for checkpoint,name in extra_weights:
            archive.write(checkpoint,name)
        archive.writestr("SUBMISSION_README.md",f"# Coordination Referee\n\nRun: `python solution.py --train train.csv --test test.csv --output predictions.csv`.\n\n{runtime_note} The archive root contains solution.py and weights/model.joblib.\n\n{training_note}\n\nBest local validation score proxy: {validation['score_proxy']}. {validation_note} {control_note}\n\nEvaluation details, other validation splits and their training sizes: validation.json.\n")
        archive.writestr("validation.json",json.dumps(validation,indent=2))
        file_count=len(archive.namelist())
    print(json.dumps({"archive":str(target),"bytes":target.stat().st_size,"files":file_count,"sha256":hashlib.sha256(target.read_bytes()).hexdigest()},indent=2))


if __name__ == "__main__":
    main()
