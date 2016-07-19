import numpy as np
import numpy.testing as npt
from sklearn.ensemble import RandomForestRegressor
import sklforestci as fci


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
    V_IJ_unbiased = fci.random_forest_error(forest, inbag, X_train, X_test)
    npt.assert_equal(V_IJ_unbiased.shape[0], y_test.shape[0])


def test_core_computation():
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

    r_vij = np.array([112.5, 387.5])

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
