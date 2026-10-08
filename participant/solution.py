#!/usr/bin/env python3
"""Шаблон решения участника — Coordination Referee.

Контракт платформы DSWorks (Contest):

    python solution.py --train train.csv --test test.csv --output predictions.csv

Этот файл — рабочий, но намеренно примитивный: он обучает модель на трёх
признаках и служит скелетом. Замените TODO своей логикой.
Полноценный пример с 122 признаками — в baseline/.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

LABELS = ["clean", "dropped_handoff", "duplicated_work", "deadlock",
          "conflict", "goal_drift", "runaway_loop"]


def parse_run(row: dict) -> dict:
    """Раскодировать JSON-поля одного рана."""
    out = {"run_id": row["run_id"], "goal": row.get("goal", "")}
    for f in ("agents", "shared_state", "messages", "artifacts", "topology"):
        v = row.get(f)
        out[f] = json.loads(v) if isinstance(v, str) else (v or [])
    return out


def featurize(run: dict) -> dict:
    """TODO: здесь ваши признаки. Ниже — три штуки для примера."""
    msgs = run["messages"]
    n = len(msgs)
    handoffs = [m for m in msgs if m.get("type") == "handoff"]
    assigns = [m for m in handoffs if not m.get("refs")]
    results = [m for m in handoffs if m.get("refs")]
    texts = [m.get("text", "") for m in msgs]
    return {
        "n_messages": n,
        "n_agents": len(run["agents"]),
        "assign_result_gap": len(assigns) - len(results),
        "dup_text_ratio": (n - len(set(texts))) / n if n else 0.0,
        "status_share": sum(1 for m in msgs if m.get("type") == "status") / n if n else 0.0,
    }


def predict_fault_turn(run: dict, label: str) -> int:
    """TODO: локализация сбоя. Возвращайте -1, если сбоя нет."""
    if label == "clean":
        return -1
    msgs = run["messages"]
    return msgs[len(msgs) // 2]["t"] if msgs else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.environ.get("TRAIN_PATH", "train.csv"))
    ap.add_argument("--test", default=os.environ.get("TEST_PATH", "test.csv"))
    ap.add_argument("--output", default=os.environ.get("OUTPUT_PATH", "predictions.csv"))
    args = ap.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    # Смоук-прогон платформы может подать урезанный файл (например, без полей
    # рана). Пишем валидный predictions.csv вместо падения.
    run_columns = ["goal", "agents", "shared_state", "messages", "artifacts",
                   "topology"]
    if any(c not in test.columns for c in run_columns) or \
            any(c not in train.columns for c in run_columns + ["label", "success"]):
        print("WARNING: входные файлы без полей рана — пишу константный ответ")
        ids = test["run_id"] if "run_id" in test.columns else range(len(test))
        pd.DataFrame({"run_id": ids, "label": "clean", "success": 1,
                      "fault_turn": -1}).to_csv(args.output, index=False)
        return

    Xtr = pd.DataFrame([featurize(parse_run(r)) for r in train.to_dict("records")])
    test_runs = [parse_run(r) for r in test.to_dict("records")]
    Xte = pd.DataFrame([featurize(r) for r in test_runs]).reindex(
        columns=Xtr.columns, fill_value=0.0)

    from sklearn.ensemble import RandomForestClassifier          # TODO: своя модель
    label_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    label_model.fit(Xtr, train["label"])
    success_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    success_model.fit(Xtr, train["success"])

    labels = list(label_model.predict(Xte))
    pd.DataFrame({
        "run_id": test["run_id"],
        "label": labels,
        "success": [int(v) for v in success_model.predict(Xte)],
        "fault_turn": [predict_fault_turn(r, l) for r, l in zip(test_runs, labels)],
    }).to_csv(args.output, index=False)
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
