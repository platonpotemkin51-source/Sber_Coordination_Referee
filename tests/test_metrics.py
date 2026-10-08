"""Metric definitions."""

from metrics import (binary_f1, composite, f1_per_class, fault_turn_hit_at_k,
                     group_f1, macro_f1)


def test_macro_f1_perfect_and_zero():
    y = ["clean", "deadlock", "conflict"]
    assert macro_f1(y, y, labels=y) == 1.0
    assert macro_f1(["clean"] * 3, ["deadlock"] * 3, labels=["clean", "deadlock"]) == 0.0


def test_macro_f1_matches_sklearn():
    sk = __import__("sklearn.metrics", fromlist=["f1_score"])
    y = ["clean", "deadlock", "conflict", "clean", "goal_drift", "deadlock"]
    p = ["clean", "conflict", "conflict", "deadlock", "goal_drift", "deadlock"]
    labels = sorted(set(y))
    assert abs(macro_f1(y, p, labels=labels)
               - sk.f1_score(y, p, average="macro", labels=labels, zero_division=0)) < 1e-9


def test_binary_f1():
    assert binary_f1([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert binary_f1([1, 1, 0, 0], [0, 0, 1, 1]) == 0.0
    assert abs(binary_f1([1, 1, 0], [1, 0, 0]) - 2 / 3) < 1e-9


def test_hit_at_2_requires_correct_class():
    lt = ["deadlock", "deadlock", "clean"]
    lp = ["deadlock", "conflict", "clean"]
    assert fault_turn_hit_at_k(lt, lp, [10, 10, -1], [12, 10, -1]) == 0.5
    assert fault_turn_hit_at_k(lt, lp, [10, 10, -1], [13, 10, -1]) == 0.0


def test_hit_at_2_ignores_clean_runs():
    assert fault_turn_hit_at_k(["clean"], ["clean"], [-1], [-1]) == 0.0


def test_f1_per_class_and_groups():
    y = ["clean", "clean", "deadlock", "deadlock"]
    p = ["clean", "deadlock", "deadlock", "deadlock"]
    per = f1_per_class(y, p, ["clean", "deadlock"])
    assert per["clean"] < per["deadlock"]
    groups = group_f1(y, p, ["a", "a", "b", "b"])
    assert groups["b"] == 1.0


def test_composite_weights_sum_to_one():
    full = {"macro_f1": 1.0, "robustness_f1": 1.0, "success_f1": 1.0,
            "fault_turn_hit2": 1.0}
    assert abs(composite(full) - 1.0) < 1e-9
    assert composite({}) == 0.0
