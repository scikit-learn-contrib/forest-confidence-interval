import copy
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
import pytest
from sklearn.ensemble import BaggingRegressor

import forestci as fci


@pytest.mark.parametrize("memory_constrained", [False, True])
@pytest.mark.parametrize("bootstrap", [False, True])
def test_calibration_reuses_supplied_inbag_for_selected_estimators(
    bootstrap, memory_constrained
):
    rng = np.random.RandomState(13)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] ** 2 + X[:, 1]
    forest = BaggingRegressor(
        n_estimators=8, max_samples=0.8, bootstrap=bootstrap, random_state=42
    ).fit(X, y)
    inbag = np.column_stack([
        np.bincount(samples, minlength=len(X))
        for samples in forest.estimators_samples_
    ])
    options = dict(memory_constrained=memory_constrained, memory_limit=0.001)
    full_variance = fci.random_forest_error(
        forest, X.shape, X[:25], inbag=inbag, calibrate=False, **options
    )
    selected = np.array([6, 1, 4, 0])
    reduced = copy.deepcopy(forest)
    reduced.estimators_ = [forest.estimators_[i] for i in selected]
    reduced._seeds = forest._seeds[selected]
    reduced.n_estimators = len(selected)
    reduced_variance = fci.random_forest_error(
        reduced, X.shape, X[:25], inbag=inbag[:, selected],
        calibrate=False, **options
    )
    expected_noise = np.mean((reduced_variance - full_variance) ** 2)
    with patch("forestci.forestci.np.random.permutation", return_value=selected), \
         patch("forestci.forestci.calc_inbag", side_effect=AssertionError("must reuse supplied counts")), \
         patch("forestci.forestci.calibrateEB", side_effect=lambda values, noise: values) as calibrate:
        result = fci.random_forest_error(
            forest, X.shape, X[:25], inbag=inbag, calibrate=True, **options
        )
    assert_allclose(result, full_variance)
    assert calibrate.call_count == 1
    assert_allclose(calibrate.call_args.args[0], full_variance)
    assert_allclose(calibrate.call_args.args[1], expected_noise)
