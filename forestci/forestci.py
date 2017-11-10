"""
Forest confidence intervals.

Calculate confidence intervals for scikit-learn RandomForestRegressor and
RandomForestClassifier predictions.
"""

import numpy as np
import copy
from .calibration import calibrateEB
from sklearn.ensemble.forest import _generate_sample_indices
from .due import _due, _BibTeX

__all__ = ("calc_inbag", "random_forest_error", "_bias_correction",
           "_core_computation")

_due.cite(_BibTeX("""
@ARTICLE{Wager2014-wn,
  title       = "Confidence Intervals for Random Forests: The Jackknife and the Infinitesimal Jackknife",
  author      = "Wager, Stefan and Hastie, Trevor and Efron, Bradley",
  journal     = "J. Mach. Learn. Res.",
  volume      =  15,
  number      =  1,
  pages       = "1625--1651",
  month       =  jan,
  year        =  2014,}"""),
          description=("Confidence Intervals for Random Forests:",
                       "The Jackknife and the Infinitesimal Jackknife"),
          path='forestci')


def calc_inbag(n_samples, forest):
    """
    Derive samples used to create trees in scikit-learn RandomForest objects.

    Recovers the samples in each tree from the random state of that tree using
    :func:`forest._generate_sample_indices`.

    Parameters
    ----------
    n_samples : int
        The number of samples used to fit the scikit-learn RandomForest object.

    forest : RandomForest
        Regressor or Classifier object that is already fit by scikit-learn.

    Returns
    -------
    Array that records how many times a data point was placed in a tree.
    Columns are individual trees. Rows are the number of times a sample was
    used in a tree.
    """

    if not forest.bootstrap:
        e_s = "Cannot calculate the inbag from a forest that has "
        e_s = " bootstrap=False"
        raise ValueError(e_s)

    n_trees = forest.n_estimators
    inbag = np.zeros((n_samples, n_trees))
    sample_idx = []
    for t_idx in range(n_trees):
        sample_idx.append(
            _generate_sample_indices(forest.estimators_[t_idx].random_state,
                                     n_samples))
        inbag[:, t_idx] = np.bincount(sample_idx[-1], minlength=n_samples)
    return inbag


def _core_computation(X_train, X_test, inbag, pred_centered, n_trees,
                      memory_constrained=False, memory_limit=None,
                      test_mode=False):
    """
    Helper function, that performs the core computation

    Parameters
    ----------
    X_train : ndarray
        An array with shape (n_train_sample, n_features).

    X_test : ndarray
        An array with shape (n_test_sample, n_features).

    inbag : ndarray
        The inbag matrix that fit the data. If set to `None` (default) it
        will be inferred from the forest. However, this only works for trees
        for which bootstrapping was set to `True`. That is, if sampling was
        done with replacement. Otherwise, users need to provide their own
        inbag matrix.

    pred_centered : ndarray
        Centered predictions that are an intermediate result in the
        computation.

    memory_constrained: boolean (optional)
        Whether or not there is a restriction on memory. If False, it is
        assumed that a ndarry of shape (n_train_sample,n_test_sample) fits
        in main memory. Setting to True can actually provide a speed up if
        memory_limit is tuned to the optimal range.

    memory_limit: int (optional)
        An upper bound for how much memory the itermediate matrices will take
        up in Megabytes. This must be provided if memory_constrained=True.


    """
    if not memory_constrained:
        return np.sum((np.dot(inbag - 1, pred_centered.T) / n_trees) ** 2, 0)

    if not memory_limit:
        raise ValueError('If memory_constrained=True, must provide',
                         'memory_limit.')

    # Assumes double precision float
    chunk_size = int((memory_limit * 1e6) / (8.0 * X_train.shape[0]))

    if chunk_size == 0:
        min_limit = 8.0 * X_train.shape[0] / 1e6
        raise ValueError('memory_limit provided is too small.' +
                         'For these dimensions, memory_limit must ' +
                         'be greater than or equal to %.3e' % min_limit)

    chunk_edges = np.arange(0, X_test.shape[0] + chunk_size, chunk_size)
    inds = range(X_test.shape[0])
    chunks = [inds[chunk_edges[i]:chunk_edges[i+1]]
              for i in range(len(chunk_edges)-1)]
    if test_mode:
        print('Number of chunks: %d' % (len(chunks),))
    V_IJ = np.concatenate([
                np.sum((np.dot(inbag-1, pred_centered[chunk].T)/n_trees)**2, 0)
                for chunk in chunks])
    return V_IJ


def _bias_correction(V_IJ, inbag, pred_centered, n_trees):
    """
    Helper functions that implements bias correction

    Parameters
    ----------
    V_IJ : ndarray
        Intermediate result in the computation.

    inbag : ndarray
        The inbag matrix that fit the data. If set to `None` (default) it
        will be inferred from the forest. However, this only works for trees
        for which bootstrapping was set to `True`. That is, if sampling was
        done with replacement. Otherwise, users need to provide their own
        inbag matrix.

    pred_centered : ndarray
        Centered predictions that are an intermediate result in the
        computation.

    n_trees : int
        The number of trees in the forest object.
    """
    n_train_samples = inbag.shape[0]
    n_var = np.mean(np.square(inbag[0:n_trees]).mean(axis=1).T.view() -
                    np.square(inbag[0:n_trees].mean(axis=1)).T.view())
    boot_var = np.square(pred_centered).sum(axis=1) / n_trees
    bias_correction = n_train_samples * n_var * boot_var / n_trees
    V_IJ_unbiased = V_IJ - bias_correction
    return V_IJ_unbiased


def random_forest_error(forest, X_train, X_test, inbag=None,
                        calibrate=True, memory_constrained=False,
                        memory_limit=None):
    """
    Calculate error bars from scikit-learn RandomForest estimators.

    RandomForest is a regressor or classifier object
    this variance can be used to plot error bars for RandomForest objects

    Parameters
    ----------
    forest : RandomForest
        Regressor or Classifier object.

    X_train : ndarray
        An array with shape (n_train_sample, n_features). The design matrix for
        training data.

    X_test : ndarray
        An array with shape (n_test_sample, n_features). The design matrix
        for testing data

    inbag : ndarray, optional
        The inbag matrix that fit the data. If set to `None` (default) it
        will be inferred from the forest. However, this only works for trees
        for which bootstrapping was set to `True`. That is, if sampling was
        done with replacement. Otherwise, users need to provide their own
        inbag matrix.

    calibrate: boolean, optional
        Whether to apply calibration to mitigate Monte Carlo noise.
        Some variance estimates may be negative due to Monte Carlo effects if
        the number of trees in the forest is too small. To use calibration,
        Default: True

    memory_constrained: boolean, optional
        Whether or not there is a restriction on memory. If False, it is
        assumed that a ndarry of shape (n_train_sample,n_test_sample) fits
        in main memory. Setting to True can actually provide a speed up if
        memory_limit is tuned to the optimal range.

    memory_limit: int, optional.
        An upper bound for how much memory the itermediate matrices will take
        up in Megabytes. This must be provided if memory_constrained=True.

    Returns
    -------
    An array with the unbiased sampling variance (V_IJ_unbiased)
    for a RandomForest object.

    See Also
    ----------
    :func:`calc_inbag`

    Notes
    -----
    The calculation of error is based on the infinitesimal jackknife variance,
    as described in [Wager2014]_ and is a Python implementation of the R code
    provided at: https://github.com/swager/randomForestCI

    .. [Wager2014] S. Wager, T. Hastie, B. Efron. "Confidence Intervals for
       Random Forests: The Jackknife and the Infinitesimal Jackknife", Journal
       of Machine Learning Research vol. 15, pp. 1625-1651, 2014.
    """
    if inbag is None:
        inbag = calc_inbag(X_train.shape[0], forest)

    pred = np.array([tree.predict(X_test) for tree in forest]).T
    pred_mean = np.mean(pred, 0)
    pred_centered = pred - pred_mean
    n_trees = forest.n_estimators
    V_IJ = _core_computation(X_train, X_test, inbag, pred_centered, n_trees,
                             memory_constrained, memory_limit)
    V_IJ_unbiased = _bias_correction(V_IJ, inbag, pred_centered, n_trees)

    # Correct for cases where resampling is done without replacement:
    if np.max(inbag) == 1:
        variance_inflation = 1 / (1 - np.mean(inbag)) ** 2
        V_IJ_unbiased *= variance_inflation

    if not calibrate:
        return V_IJ_unbiased

    if V_IJ_unbiased.shape[0] <= 20:
        print("No calibration with n_samples <= 20")
        return V_IJ_unbiased
    if calibrate:

        calibration_ratio = 2
        n_sample = np.ceil(n_trees / calibration_ratio)
        new_forest = copy.deepcopy(forest)
        new_forest.estimators_ =\
            np.random.permutation(new_forest.estimators_)[:int(n_sample)]
        new_forest.n_estimators = int(n_sample)

        results_ss = random_forest_error(new_forest, X_train, X_test,
                                         calibrate=False)
        # Use this second set of variance estimates
        # to estimate scale of Monte Carlo noise
        sigma2_ss = np.mean((results_ss - V_IJ_unbiased)**2)
        delta = n_sample / n_trees
        sigma2 = (delta**2 + (1 - delta)**2) / (2 * (1 - delta)**2) * sigma2_ss

        # Use Monte Carlo noise scale estimate for empirical Bayes calibration
        V_IJ_calibrated = calibrateEB(V_IJ_unbiased, sigma2)

        return V_IJ_calibrated
