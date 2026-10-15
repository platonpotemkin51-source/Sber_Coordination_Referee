"""Competition entry point for the best locally validated v15 ensemble."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from v15_sequence_inference import infer_candidate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Accepted for the competition run contract; frozen models do not retrain.")
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    from referee.data import load_data
    from referee.relations import relation_features

    base = ROOT / "weights"
    bundle = joblib.load(base / "model.joblib")
    ranker = joblib.load(ROOT / "outputs/stack_v9/model.joblib")["ranker"]
    data = relation_features(load_data(args.test), args.test)
    prediction = infer_candidate(
        bundle,
        ranker,
        data,
        base,
        ROOT / "outputs/sequence_v6/model.pt",
        args.device,
        ROOT / "weights/word-char-model.joblib",
    )
    prediction.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(prediction)} rows", flush=True)


if __name__ == "__main__":
    main()
