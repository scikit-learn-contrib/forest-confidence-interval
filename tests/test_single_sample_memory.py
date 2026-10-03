import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.ensemble import RandomForestRegressor

import forestci as fci


@pytest.mark.parametrize("memory_limit", [0.00032, 1])
@pytest.mark.parametrize("calibrate", [False, True])
def test_single_sample_vector_matches_matrix_in_memory_modes(memory_limit, calibrate):
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] - 2 * X[:, 1]
    forest = RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)
    expected = fci.random_forest_error(
        forest, X.shape, X[:1], calibrate=calibrate
    )
    actual = fci.random_forest_error(
        forest, X.shape, X[0], calibrate=calibrate,
        memory_constrained=True, memory_limit=memory_limit,
    )
    assert actual.shape == (1,)
    assert_allclose(actual, expected)


def test_single_feature_batch_keeps_each_sample_in_memory_mode():
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = np.sin(X[:, 0])
    forest = RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)
    expected = fci.random_forest_error(forest, X.shape, X[:5], calibrate=False)
    actual = fci.random_forest_error(
        forest, X.shape, X[:5], calibrate=False,
        memory_constrained=True, memory_limit=0.00032,
    )
    assert actual.shape == (5,)
    assert_allclose(actual, expected)
