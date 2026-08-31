from inspect import signature
import sys
from unittest.mock import patch
import pytest

import forestci as fci
import forestci.calibration as calib
import numpy as np
import numpy.testing as npt

from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def test_random_forest_error():
    X = np.array([[5, 2], [5, 5], [3, 3], [6, 4], [6, 6]])

    y = np.array([70, 100, 60, 100, 120])

    train_idx = [2, 3, 4]
    test_idx = [0, 1]

    y_test = y[test_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    X_train = X[train_idx]

    n_trees = 4
    forest = RandomForestRegressor(n_estimators=n_trees, random_state=42)
    forest.fit(X_train, y_train)
    inbag = fci.calc_inbag(X_train.shape[0], forest)
    for ib in [inbag, None]:
        for calibrate in [True, False]:
            V_IJ_unbiased = fci.random_forest_error(
                forest, X_train.shape, X_test, inbag=ib, calibrate=calibrate
            )
        npt.assert_equal(V_IJ_unbiased.shape[0], y_test.shape[0])

    # We cannot calculate inbag from a non-bootstrapped forest. This is because
    # Scikit-learn trees do not store their own sample weights. If you did This
    # some other way, you can still use your own inbag
    non_bootstrap_forest = RandomForestRegressor(
        n_estimators=n_trees, bootstrap=False, random_state=42
    )

    npt.assert_raises(
        ValueError, fci.calc_inbag, X_train.shape[0], non_bootstrap_forest
    )


def test_random_forest_error_multioutput():
    X = np.array([[5, 2], [5, 5], [3, 3], [6, 4], [6, 6]])

    y = np.array([[70, 37], [100, 55], [60, 33], [100, 54], [120, 66]])

    train_idx = [2, 3, 4]
    test_idx = [0, 1]

    y_test = y[test_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    X_train = X[train_idx]

    n_trees = 4
    forest = RandomForestRegressor(n_estimators=n_trees, random_state=42)
    forest.fit(X_train, y_train)

    V_IJ_unbiased_target0 = fci.random_forest_error(
        forest, X_train.shape, X_test, calibrate=True, y_output=0
    )
    npt.assert_equal(V_IJ_unbiased_target0.shape[0], y_test.shape[0])

    # With a MultiOutput RandomForestRegressor the user MUST specify a y_output
    npt.assert_raises(
        ValueError,
        fci.random_forest_error,
        forest,
        X_train.shape,
        X_test,
        inbag=None,
        calibrate=True,
        memory_constrained=False,
        memory_limit=None,
        y_output=None,  # This should trigger the ValueError
    )


def test_bagging_svr_error():
    X = np.array([[5, 2], [5, 5], [3, 3], [6, 4], [6, 6]])

    y = np.array([70, 100, 60, 100, 120])

    train_idx = [2, 3, 4]
    test_idx = [0, 1]

    y_test = y[test_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    X_train = X[train_idx]

    n_trees = 4
    estimator_parameter = (
        "estimator"
        if "estimator" in signature(BaggingRegressor).parameters
        else "base_estimator"
    )
    bagger = BaggingRegressor(
        **{
            estimator_parameter: SVR(),
            "n_estimators": n_trees,
            "random_state": 42,
        }
    )
    bagger.fit(X_train, y_train)
    inbag = fci.calc_inbag(X_train.shape[0], bagger)
    for ib in [inbag, None]:
        for calibrate in [True, False]:
            V_IJ_unbiased = fci.random_forest_error(
                bagger, X_train.shape, X_test, inbag=ib, calibrate=calibrate
            )
        npt.assert_equal(V_IJ_unbiased.shape[0], y_test.shape[0])


def test_core_computation():
    inbag_ex = np.array(
        [[1.0, 2.0, 0.0, 1.0], [1.0, 0.0, 2.0, 0.0], [1.0, 1.0, 1.0, 2.0]]
    )

    X_train_ex = np.array([[3, 3], [6, 4], [6, 6]])
    X_test_ex = np.vstack([np.array([[5, 2], [5, 5]]) for _ in range(1000)])
    pred_centered_ex = np.vstack(
        [
            np.array([[-20, -20, 10, 30], [-20, 30, -20, 10]])
            for _ in range(1000)
        ]
    )
    n_trees = 4

    our_vij = fci._core_computation(
        X_train_ex.shape, X_test_ex, inbag_ex, pred_centered_ex, n_trees
    )

    r_vij = np.concatenate([np.array([112.5, 387.5]) for _ in range(1000)])

    npt.assert_almost_equal(our_vij, r_vij)

    for memory_constrained, memory_limit in zip([True, False], [0.01, None]):
        our_vij = fci._core_computation(
            X_train_ex.shape,
            X_test_ex,
            inbag_ex,
            pred_centered_ex,
            n_trees,
            memory_constrained=memory_constrained,
            memory_limit=memory_limit,
            test_mode=True,
        )

        npt.assert_almost_equal(our_vij, r_vij)


def test_bias_correction():
    inbag_ex = np.array(
        [[1.0, 2.0, 0.0, 1.0], [1.0, 0.0, 2.0, 0.0], [1.0, 1.0, 1.0, 2.0]]
    )

    X_train_ex = np.array([[3, 3], [6, 4], [6, 6]])

    X_test_ex = np.array([[5, 2], [5, 5]])

    pred_centered_ex = np.array([[-20, -20, 10, 30], [-20, 30, -20, 10]])
    n_trees = 4

    our_vij = fci._core_computation(
        X_train_ex.shape, X_test_ex, inbag_ex, pred_centered_ex, n_trees
    )
    our_vij_unbiased = fci._bias_correction(
        our_vij, inbag_ex, pred_centered_ex, n_trees
    )
    r_unbiased_vij = np.array([-42.1875, 232.8125])
    npt.assert_almost_equal(our_vij_unbiased, r_unbiased_vij)


def test_with_calibration():
    # Test both with and without interpolation:
    for n in [25 * 5, 205 * 5]:
        random_state = np.random.RandomState(42)
        X = random_state.rand(n).reshape(n // 5, 5)
        y = random_state.rand(n // 5)

        train_idx = np.arange(int(n // 5 * 0.75))
        test_idx = np.arange(int(n // 5 * 0.75), n // 5)

        y_test = y[test_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        X_train = X[train_idx]

        n_trees = 4
        forest = RandomForestRegressor(n_estimators=n_trees, random_state=42)
        forest.fit(X_train, y_train)
        V_IJ_unbiased = fci.random_forest_error(forest, X_train.shape, X_test)
        npt.assert_equal(V_IJ_unbiased.shape[0], y_test.shape[0])


def test_centered_prediction_forest():
    X = np.array([[5, 2], [5, 5], [3, 3], [6, 4], [6, 6]])

    y = np.array([70, 100, 60, 100, 120])

    train_idx = [2, 3, 4]
    test_idx = [0, 1]

    y_train = y[train_idx]
    X_test = X[test_idx]
    X_train = X[train_idx]

    n_trees = 8
    forest = RandomForestRegressor(n_estimators=n_trees, random_state=42)
    forest = forest.fit(X_train, y_train)

    # test different amount of test samples
    for i in range(len(X_test)):
        test_samples = X_test[: i + 1]
        pred_centered = fci.forestci._centered_prediction_forest(
            forest, test_samples
        )

        # the vectorized solution has to match the single sample predictions
        for n_sample, sample in enumerate(test_samples):
            # This assignment assures correctness of single-sample
            # calculations. No extra averaging test is required because
            # for a single sample
            # dimension 0 (i.e. the number of test sets) disappears
            pred_centered_sample = fci.forestci._centered_prediction_forest(
                forest, sample
            )
            assert len(pred_centered_sample[0]) == n_trees
            npt.assert_almost_equal(
                pred_centered_sample[0],
                pred_centered[n_sample],
            )


def test_classifier_calibration_inflation():
    X, y = make_classification(n_samples=500, n_features=6, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    X_test = X[:100]
    np.random.seed(0)
    uncalibrated = fci.random_forest_error(clf, X.shape, X_test, calibrate=False)
    assert np.any(uncalibrated <= 0)
    np.random.seed(0)
    calibrated = fci.random_forest_error(clf, X.shape, X_test, calibrate=True)
    assert calibrated[uncalibrated <= 0].mean() < uncalibrated.mean()


def test_gfit_negative_support():
    X = np.array([-0.8, -0.3, 0.1, 0.4, 0.9, 1.2])
    xvals, g_eta = calib.gfit(X, sigma=0.1)
    assert np.min(xvals) < 0
    assert len(xvals) == 1000


@pytest.mark.parametrize(
    "variances",
    [
        np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
        np.linspace(-1.0, 1.0, 101),
        np.array([-0.5, -0.4, -0.3, -0.2, -0.1]),
    ],
)
def test_calibrateEB_is_finite_for_negative_and_symmetric_inputs(variances):
    calibrated = calib.calibrateEB(variances, sigma2=0.01)
    assert np.all(np.isfinite(calibrated))
    assert np.all(calibrated >= 0)


@pytest.mark.parametrize("value", [0.0, -0.1])
def test_gfit_constant_nonpositive_input(value):
    xvals, prior = calib.gfit(np.full(5, value), sigma=0.1)
    assert np.all(np.isfinite(prior))
    npt.assert_allclose(prior.sum(), 1.0)
    assert np.all(prior[xvals < 0] == 0)


def test_show_progress():
    X = np.array([[5, 2], [5, 5], [3, 3], [6, 4], [6, 6]])
    y = np.array([70, 100, 60, 100, 120])

    n_trees = 4
    forest = RandomForestRegressor(n_estimators=n_trees, random_state=42)
    forest.fit(X, y)

    with patch(
        "tqdm.auto.tqdm", side_effect=lambda x, **kwargs: x
    ) as mock_tqdm:
        err_progress = fci.random_forest_error(
            forest,
            X.shape,
            X,
            calibrate=False,
            memory_constrained=True,
            memory_limit=0.00008,
            show_progress=True,
        )
        mock_tqdm.assert_called_once()
        assert len(mock_tqdm.call_args.args[0]) > 1
        assert mock_tqdm.call_args.kwargs["desc"] == "Computing V_IJ (4 trees)"

    err_no_progress = fci.random_forest_error(
        forest,
        X.shape,
        X,
        calibrate=False,
        memory_constrained=True,
        memory_limit=0.00008,
        show_progress=False,
    )

    npt.assert_almost_equal(err_progress, err_no_progress)


def test_show_progress_without_tqdm():
    X = np.array([[5, 2], [5, 5], [3, 3], [6, 4], [6, 6]])
    y = np.array([70, 100, 60, 100, 120])
    forest = RandomForestRegressor(n_estimators=4, random_state=42).fit(X, y)

    with patch.dict(sys.modules, {"tqdm.auto": None}):
        with pytest.warns(UserWarning, match="tqdm must be installed"):
            err_without_tqdm = fci.random_forest_error(
                forest,
                X.shape,
                X,
                calibrate=False,
                memory_constrained=True,
                memory_limit=0.00008,
                show_progress=True,
            )

    err_no_progress = fci.random_forest_error(
        forest,
        X.shape,
        X,
        calibrate=False,
        memory_constrained=True,
        memory_limit=0.00008,
    )
    npt.assert_almost_equal(err_without_tqdm, err_no_progress)


def test_calibration_progress_labels_tree_counts():
    random_state = np.random.RandomState(42)
    X = random_state.normal(size=(30, 2))
    y = random_state.normal(size=30)
    forest = RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)

    with (
        patch(
            "tqdm.auto.tqdm", side_effect=lambda x, **kwargs: x
        ) as mock_tqdm,
        patch(
            "forestci.forestci.calibrateEB",
            side_effect=lambda variances, _: variances,
        ),
    ):
        fci.random_forest_error(
            forest,
            X.shape,
            X,
            memory_constrained=True,
            memory_limit=0.001,
            show_progress=True,
        )

    assert mock_tqdm.call_count == 2
    descriptions = [call.kwargs["desc"] for call in mock_tqdm.call_args_list]
    assert descriptions == [
        "Computing V_IJ (20 trees)",
        "Computing V_IJ (10 trees)",
    ]
