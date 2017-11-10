import numpy as np
import numpy.testing as npt
from sklearn.ensemble import RandomForestRegressor
import forestci as fci


def test_random_forest_error():
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

    n_trees = 4
    forest = RandomForestRegressor(n_estimators=n_trees)
    forest.fit(X_train, y_train)
    inbag = fci.calc_inbag(X_train.shape[0], forest)
    for ib in [inbag, None]:
        for calibrate in [True, False]:
            V_IJ_unbiased = fci.random_forest_error(forest, X_train, X_test,
                                                    inbag=ib,
                                                    calibrate=calibrate)
        npt.assert_equal(V_IJ_unbiased.shape[0], y_test.shape[0])

    # We cannot calculate inbag from a non-bootstrapped forest. This is because
    # Scikit-learn trees do not store their own sample weights. If you did This
    # some other way, you can still use your own inbag
    non_bootstrap_forest = RandomForestRegressor(n_estimators=n_trees,
                                                 bootstrap=False)

    npt.assert_raises(ValueError, fci.calc_inbag, X_train.shape[0],
                      non_bootstrap_forest)


def test_core_computation():
    inbag_ex = np.array([[1., 2., 0., 1.],
                         [1., 0., 2., 0.],
                         [1., 1., 1., 2.]])

    X_train_ex = np.array([[3, 3],
                           [6, 4],
                           [6, 6]])
    X_test_ex = np.vstack([np.array([[5, 2],
                                     [5, 5]]) for _ in range(1000)])
    pred_centered_ex = np.vstack([np.array([[-20, -20, 10, 30],
                                            [-20, 30, -20, 10]])
                                  for _ in range(1000)])
    n_trees = 4

    our_vij = fci._core_computation(X_train_ex, X_test_ex, inbag_ex,
                                    pred_centered_ex, n_trees)

    r_vij = np.concatenate([np.array([112.5, 387.5]) for _ in range(1000)])

    npt.assert_almost_equal(our_vij, r_vij)

    for mc, ml in zip([True, False], [.01, None]):
        our_vij = fci._core_computation(X_train_ex, X_test_ex, inbag_ex,
                                        pred_centered_ex, n_trees,
                                        memory_constrained=True,
                                        memory_limit=.01,
                                        test_mode=True)

        npt.assert_almost_equal(our_vij, r_vij)


def test_bias_correction():
    inbag_ex = np.array([[1., 2., 0., 1.],
                         [1., 0., 2., 0.],
                         [1., 1., 1., 2.]])

    X_train_ex = np.array([[3, 3],
                           [6, 4],
                           [6, 6]])

    X_test_ex = np.array([[5, 2],
                          [5, 5]])

    pred_centered_ex = np.array([[-20, -20, 10, 30], [-20, 30, -20, 10]])
    n_trees = 4

    our_vij = fci._core_computation(X_train_ex, X_test_ex, inbag_ex,
                                    pred_centered_ex, n_trees)
    our_vij_unbiased = fci._bias_correction(our_vij, inbag_ex,
                                            pred_centered_ex,
                                            n_trees)
    r_unbiased_vij = np.array([-42.1875, 232.8125])
    npt.assert_almost_equal(our_vij_unbiased, r_unbiased_vij)


def test_with_calibration():
    # Test both with and without interpolation:
    for n in [25 * 5, 205 * 5]:
        X = np.random.rand(n).reshape(n // 5, 5)
        y = np.random.rand(n // 5)

        train_idx = np.arange(int(n // 5 * 0.75))
        test_idx = np.arange(int(n//5 * 0.75), n//5)

        y_test = y[test_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        X_train = X[train_idx]

        n_trees = 4
        forest = RandomForestRegressor(n_estimators=n_trees)
        forest.fit(X_train, y_train)
        V_IJ_unbiased = fci.random_forest_error(forest, X_train, X_test)
        npt.assert_equal(V_IJ_unbiased.shape[0], y_test.shape[0])
