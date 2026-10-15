"""Competition entry point for the best locally validated v15 ensemble."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Bound native CPU pools before importing numerical libraries.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
RUN_COLUMNS = ("goal", "agents", "shared_state", "messages", "artifacts", "topology")
OUTPUT_COLUMNS = ["run_id", "label", "success", "fault_turn"]


def has_messages(value) -> bool:
    if pd.isna(value) or not str(value).strip():
        return False
    messages = json.loads(value)
    if messages is None:
        return False
    if not isinstance(messages, list):
        raise ValueError("messages must contain a JSON list")
    return bool(messages)


def predict_frame(test, source, device):
    """Honor the platform's reduced-input contract before loading any weights."""
    if "run_id" not in test:
        raise ValueError("test.csv must contain run_id")
    result = pd.DataFrame({"run_id": test["run_id"], "label": "clean",
                           "success": 1, "fault_turn": -1})
    missing = [name for name in RUN_COLUMNS if name not in test]
    if missing or test.empty:
        print(f"Reduced-input fallback: {len(test)} rows; missing fields: {missing}", flush=True)
        return result

    active = test["messages"].map(has_messages)
    fallback_count = int((~active).sum())
    if fallback_count:
        print(f"No-message fallback: {fallback_count}/{len(test)} rows", flush=True)
    if not active.any():
        return result

    # Initialize Torch's native runtime before loading LightGBM/OpenMP models.
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        use_neural = False
        print("PyTorch unavailable: using frozen tree/ranker/text models and tree success", flush=True)
    else:
        torch.set_num_threads(4)
        use_neural = True
    import joblib
    from threadpoolctl import threadpool_limits
    from v15_sequence_inference import infer_candidate
    from referee.data import load_data
    from referee.relations import relation_features

    started = time.monotonic()
    print(f"Loading frozen models for {int(active.sum())} runs", flush=True)
    base = ROOT / "weights"
    bundle = joblib.load(base / "model.joblib")
    ranker = joblib.load(ROOT / "outputs/stack_v9/model.joblib")["ranker"]
    # Keep caches private to this invocation, including on repeated platform runs.
    with tempfile.TemporaryDirectory(prefix="referee-inference-") as temporary:
        source_path = Path(source)
        if fallback_count:
            source_path = Path(temporary) / "active.csv"
            test.loc[active].to_csv(source_path, index=False)
        data = load_data(source_path, cache=Path(temporary) / "features")
        data = relation_features(data, source_path, cache=Path(temporary) / "relations")
        data["run_id"] = test.loc[active, "run_id"].to_numpy()
        print(f"Features ready in {time.monotonic() - started:.1f}s", flush=True)
        with threadpool_limits(limits=4):
            prediction = infer_candidate(
                bundle, ranker, data, base, ROOT / "outputs/sequence_v6/model.pt",
                device, ROOT / "weights/word-char-model.joblib", use_neural=use_neural,
            )
    result.loc[active, OUTPUT_COLUMNS] = prediction[OUTPUT_COLUMNS].to_numpy()
    result[["success", "fault_turn"]] = result[["success", "fault_turn"]].astype(int)
    print(f"Model inference completed in {time.monotonic() - started:.1f}s", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Accepted for the competition run contract; frozen models do not retrain.")
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    test = pd.read_csv(args.test, dtype={"run_id": str}, keep_default_na=False)
    prediction = predict_frame(test, args.test, args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(prediction)} rows", flush=True)


if __name__ == "__main__":
    main()
