"""Target independence and probability mass conservation for pair attribution."""
import copy

import numpy as np
import pandas as pd

from referee.pair_selector import PairSelector, pair_features, redistribute_pair, top_two


def observed_data():
    return {
        "X": pd.DataFrame({"n_messages": [3., 2.], "observable_count": [1., 2.]}),
        "turn_X": np.arange(15, dtype=np.float32).reshape(5, 3),
        "turn_columns": ["assignment", "pending_delivery", "artifact_count"],
        "offsets": np.array([0, 3, 5]), "y": np.array([1, 2]),
        "success": np.array([0, 1]), "fault": np.array([1, 0]),
        "run_id": np.array(["opaque_a", "opaque_b"]),
    }


def detector_outputs():
    p = np.array([[.05, .45, .3, .05, .05, .05, .05], [.1, .1, .5, .1, .05, .1, .05]])
    evidence = np.array([[.1, .9, .8, .2, .1, .1, .1], [.2, .3, .9, .2, .1, .3, .1]])
    candidates = np.array([[-1, 1, 2, 0, 1, 1, 2], [-1, 0, 0, 1, 1, 0, 0]])
    return p, evidence, candidates


def test_targets_and_run_identifiers_do_not_change_pair_features():
    data = observed_data()
    changed = copy.deepcopy(data)
    changed["y"][:] = 6
    changed["success"][:] = 0
    changed["fault"][:] = 999
    changed["run_id"][:] = "other"
    indexes = np.array([0, 1])
    original = pair_features(data, indexes, *detector_outputs())
    mutated = pair_features(changed, indexes, *detector_outputs())
    for left, right in zip(original, mutated):
        np.testing.assert_array_equal(left, right)


def test_pair_probability_preserves_mass_and_other_classes():
    original = np.array([[.15, .5, .2, .05, .04, .03, .03], [.5, .2, .1, .05, .05, .05, .05]])
    pairs = top_two(original)
    result = redistribute_pair(original, pairs, np.array([.8, .25]))
    assert result.shape == (2, 7)
    np.testing.assert_allclose(result.sum(1), 1, rtol=0, atol=1e-12)
    for row in range(2):
        other = [label for label in range(7) if label not in pairs[row]]
        np.testing.assert_array_equal(result[row, other], original[row, other])
    np.testing.assert_allclose(result[0, [1, 2]], [.14, .56], rtol=0, atol=1e-12)


def test_predict_keeps_pair_mapping_when_features_are_reordered():
    class FrozenBinaryClassifier:
        def predict_proba(self, features):
            return np.tile([.3, .7], (len(features), 1))

    data = observed_data()
    selector = PairSelector()
    selector.run_columns = list(data["X"].columns)
    selector.turn_columns = list(data["turn_columns"])
    selector.classifier = FrozenBinaryClassifier()
    expected = selector.predict(data, np.array([0, 1]), *detector_outputs())
    reordered = dict(data)
    reordered["X"] = data["X"][list(reversed(data["X"].columns))]
    reordered["turn_X"] = data["turn_X"][:, ::-1]
    reordered["turn_columns"] = list(reversed(data["turn_columns"]))
    actual = selector.predict(reordered, np.array([0, 1]), *detector_outputs())
    np.testing.assert_array_equal(expected, actual)
    np.testing.assert_allclose(actual.sum(1), 1, rtol=0, atol=1e-12)
