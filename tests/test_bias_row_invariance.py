"""Regression tests for training-row truncation in the IJ bias correction."""
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest
from sklearn.ensemble import RandomForestRegressor

import forestci.forestci as fci


def _all_rows_reference(vij, inbag, pred_centered, n_trees):
    count_variance = np.var(inbag, axis=1, ddof=0).mean()
    prediction_variance = np.mean(pred_centered**2, axis=1)
    return vij - len(inbag) * count_variance * prediction_variance / n_trees


def test_bias_correction_uses_every_training_row():
    # Valid bootstrap counts: each column sums to the four training rows.
    inbag = np.array([[4., 0.], [0., 4.], [0., 0.], [0., 0.]])
    pred_centered = np.array([[-1., 1.]])
    for order in [np.arange(4), np.array([2, 3, 0, 1])]:
        counts = inbag[order]
        raw = fci._core_computation(
            (4, 1), np.zeros((1, 1)), counts, pred_centered, 2
        )
        npt.assert_allclose(raw, [8.])
        corrected = fci._bias_correction(raw, counts, pred_centered, 2)
        npt.assert_allclose(corrected, [4.])


@pytest.mark.parametrize('n_rows,n_trees', [(50, 10), (50, 50), (50, 80)])
def test_bias_correction_matches_all_row_population_variance(n_rows, n_trees):
    rng = np.random.default_rng(903)
    inbag = rng.multinomial(
        n_rows, np.full(n_rows, 1 / n_rows), size=n_trees
    ).T
    pred = rng.normal(size=(7, n_trees))
    pred -= pred.mean(axis=1, keepdims=True)
    raw = fci._core_computation(
        (n_rows, 1), np.zeros((7, 1)), inbag, pred, n_trees
    )
    expected = _all_rows_reference(raw, inbag, pred, n_trees)
    for order in [np.arange(n_rows), rng.permutation(n_rows)]:
        corrected = fci._bias_correction(raw, inbag[order], pred, n_trees)
        npt.assert_allclose(corrected, expected)


@pytest.mark.parametrize('memory_constrained', [False, True])
def test_public_api_is_invariant_to_inbag_row_relabeling(memory_constrained):
    rng = np.random.default_rng(121)
    x = rng.normal(size=(60, 3))
    y = x[:, 0] + rng.normal(size=60)
    xt = rng.normal(size=(25, 3))
    forest = RandomForestRegressor(n_estimators=20, random_state=42).fit(x, y)
    inbag = fci.calc_inbag(len(x), forest)
    kwargs = dict(
        calibrate=False,
        memory_constrained=memory_constrained,
        memory_limit=.002 if memory_constrained else None,
    )
    baseline = fci.random_forest_error(
        forest, x.shape, xt, inbag=inbag, **kwargs
    )
    permuted = fci.random_forest_error(
        forest, x.shape, xt, inbag=inbag[rng.permutation(len(x))], **kwargs
    )
    npt.assert_allclose(baseline, permuted, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('n_trees', [30, 45, 59, 60])
def test_calibration_half_forest_uses_every_training_row(n_trees):
    # The calibration half-forest can have fewer trees than training rows,
    # even when the full forest has at least as many trees as training rows.
    rng = np.random.default_rng(222)
    x = rng.normal(size=(30, 3))
    y = rng.normal(size=30)
    xt = rng.normal(size=(25, 3))
    forest = RandomForestRegressor(n_estimators=n_trees, random_state=12)
    forest.fit(x, y)
    original = fci._bias_correction
    calls = []

    def checked_correction(vij, inbag, pred_centered, b):
        result = original(vij, inbag, pred_centered, b)
        expected = _all_rows_reference(vij, inbag, pred_centered, b)
        calls.append(b)
        npt.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
        return result

    # Calibration samples trees using NumPy's global RNG; restore its state
    # so the deterministic subsample does not affect unrelated tests.
    random_state = np.random.get_state()
    try:
        np.random.seed(85)
        with patch.object(fci, '_bias_correction',
                          side_effect=checked_correction), \
                patch.object(fci, 'calibrateEB',
                             side_effect=lambda variances, _: variances):
            result = fci.random_forest_error(
                forest, x.shape, xt, calibrate=True
            )
    finally:
        np.random.set_state(random_state)

    assert calls == [n_trees, int(np.ceil(n_trees / 2))]
    assert result.shape == (len(xt),)
