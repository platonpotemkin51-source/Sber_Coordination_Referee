"""Package the frozen v15 ensemble plus validated v9/sequence/word-char components."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "handoff/v15/assembled"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/submission-v15-runtime-fixed-20261015.zip")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    original = HANDOFF / "submission-v15-score092.zip"
    text_model = ROOT / "outputs/char-tfidf-v15-experiment/model.joblib"
    ranker = HANDOFF / "outputs/stack_v9/model.joblib"
    sequence = HANDOFF / "outputs/sequence_v6/model.pt"
    metrics = json.loads((ROOT / "outputs/char-tfidf-v15-experiment/metrics.json").read_text())
    candidate_metrics = {
        "candidate": "frozen v15 + v9 + sequence_v6 + word_char",
        "local_score_proxy": {
            split: metrics["word_char"][key]
            for split, key in (("development", "development"), ("check_half", "check"), ("fold0", "fold0"))
        },
        "word_char_class_gates": metrics["word_char"]["class_gates"],
        "label_order": ["clean", "dropped_handoff", "duplicated_work", "deadlock", "conflict", "goal_drift", "runaway_loop"],
        "note": "All values are local score proxies, not an official leaderboard result. Check-half and fold 0 were used during development.",
    }
    replacement_solution = (ROOT / "scripts/candidate_solution.py").read_bytes()
    wrapper = (ROOT / "scripts/v15_sequence_inference.py").read_bytes()
    readme = """# Coordination Referee — v15 word+char candidate

Run with the competition contract: `python solution.py --train train.csv --test test.csv --output predictions.csv`.

This archive keeps the colleague's frozen v15 weights and adds its validated v9 ranker, sequence_v6 class checkpoint, and word+char TF-IDF model. Models are frozen; inference does not train or use network access. PyTorch 2.5.1 is required by the sequence component; CPU inference is the default and `--device cuda` is optional.

Choose the platform GPU image (includes PyTorch 2.5.1) for the complete ensemble, even with default CPU inference. If PyTorch is absent, the entry point uses frozen tree, selector, ranker and text models with tree success; the local metrics below do not describe that reduced ensemble. Reduced smoke inputs without run columns produce clean/1/-1 before model imports, as specified by the competition FAQ. Individual runs without messages use the same fallback. sklearn 1.5.2 compatibility is handled for the frozen multinomial logistic model.

The best local score proxies are 0.919870 development, 0.910097 check-half, and 0.913721 fold 0. They are not official platform scores; the check-half and fold 0 have informed model development. See `candidate_metrics.json` for component details and `validation.json` for the original v15 validation record.
"""
    with ZipFile(original) as source, ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as target:
        replaced = {"solution.py", "SUBMISSION_README.md", "referee/data.py",
                    "referee/relations.py", "referee/model.py", "referee/neural_success.py"}
        for item in source.infolist():
            if item.filename not in replaced:
                target.writestr(item, source.read(item.filename))
        target.writestr("solution.py", replacement_solution)
        target.writestr("scripts/v15_sequence_inference.py", wrapper)
        for module in ("data.py", "relations.py", "model.py", "stack.py", "neural_success.py"):
            module_path = ROOT / "referee" / module
            if not module_path.exists():
                module_path = HANDOFF / "referee" / module
            target.write(module_path, "referee/" + module)
        target.write(ranker, "outputs/stack_v9/model.joblib")
        target.write(sequence, "outputs/sequence_v6/model.pt")
        target.write(text_model, "weights/word-char-model.joblib")
        target.writestr("SUBMISSION_README.md", readme)
        target.writestr("candidate_metrics.json", json.dumps(candidate_metrics, indent=2))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    with ZipFile(output) as archive:
        names = archive.namelist()
        duplicates = len(names) - len(set(names))
    if duplicates:
        raise RuntimeError(f"Candidate archive contains {duplicates} duplicate member names")
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size,
                      "sha256": digest, "files": len(names), "metrics": candidate_metrics}, indent=2))


if __name__ == "__main__":
    main()
