import numpy as np
import numpy.testing as npt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
import forestci as fci


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
    forest = RandomForestRegressor(n_estimators=n_trees)
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
    non_bootstrap_forest = RandomForestRegressor(n_estimators=n_trees, bootstrap=False)

    npt.assert_raises(
        ValueError, fci.calc_inbag, X_train.shape[0], non_bootstrap_forest
    )


def test_random_forest_error_multioutput():
    X = np.array([[5, 2], [5, 5], [3, 3], [6, 4], [6, 6]])

    y = np.array([[70, 37], [100, 55], [60, 33], [100,54], [120, 66]])

    train_idx = [2, 3, 4]
    test_idx = [0, 1]

    y_test = y[test_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    X_train = X[train_idx]

    n_trees = 4
    forest = RandomForestRegressor(n_estimators=n_trees)
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
        y_output=None # This should trigger the ValueError
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
    bagger = BaggingRegressor(estimator=SVR(), n_estimators=n_trees)
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
        [np.array([[-20, -20, 10, 30], [-20, 30, -20, 10]]) for _ in range(1000)]
    )
    n_trees = 4

    our_vij = fci._core_computation(
        X_train_ex.shape, X_test_ex, inbag_ex, pred_centered_ex, n_trees
    )

    r_vij = np.concatenate([np.array([112.5, 387.5]) for _ in range(1000)])

    npt.assert_almost_equal(our_vij, r_vij)

    for mc, ml in zip([True, False], [0.01, None]):
        our_vij = fci._core_computation(
            X_train_ex.shape,
            X_test_ex,
            inbag_ex,
            pred_centered_ex,
            n_trees,
            memory_constrained=True,
            memory_limit=0.01,
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
        X = np.random.rand(n).reshape(n // 5, 5)
        y = np.random.rand(n // 5)

        train_idx = np.arange(int(n // 5 * 0.75))
        test_idx = np.arange(int(n // 5 * 0.75), n // 5)

        y_test = y[test_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        X_train = X[train_idx]

        n_trees = 4
        forest = RandomForestRegressor(n_estimators=n_trees)
        forest.fit(X_train, y_train)
        V_IJ_unbiased = fci.random_forest_error(forest, X_train.shape, X_test)
        npt.assert_equal(V_IJ_unbiased.shape[0], y_test.shape[0])


def test_centered_prediction_forest():
    X = np.array([[5, 2],
                  [5, 5],
                  [3, 3],
                  [6, 4],
                  [6, 6]])

    y = np.array([70, 100, 60, 100, 120])

    train_idx = [2, 3, 4]
    test_idx = [0, 1]

    y_test = y[test_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    X_train = X[train_idx]

    n_trees = 8
    forest = RandomForestRegressor(n_estimators=n_trees)
    forest = forest.fit(X_train, y_train)

    # test different amount of test samples
    for i in range(len(X_test)):
        test_samples = X_test[:i+1]
        pred_centered = fci.forestci._centered_prediction_forest(forest, test_samples)

        # the vectorized solution has to match the single sample predictions
        for n_sample, sample in enumerate(test_samples):
            # the following assignment assures correctness of single test sample calculations
            # no additional tests for correct averaging required since for single test samples
            # dimension 0 (i.e. the number of test sets) disappears
            pred_centered_sample = fci.forestci._centered_prediction_forest(
                forest, sample)
            assert len(pred_centered_sample[0]) == n_trees
            npt.assert_almost_equal(
                pred_centered_sample[0],
                pred_centered[n_sample]
                )
