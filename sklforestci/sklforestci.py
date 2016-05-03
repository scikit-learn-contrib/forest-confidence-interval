import numpy as np
from sklearn.ensemble.forest import _generate_sample_indices
from .due import due, BibTeX

__all__ = ["calc_inbag", "random_forest_error", "_bias_correction",
           "_core_computation"]

due.cite(BibTeX("""
@ARTICLE{Wager2014-wn,
  title       = "Confidence Intervals for Random Forests: The Jackknife and the
                 Infinitesimal Jackknife",
  author      = "Wager, Stefan and Hastie, Trevor and Efron, Bradley",
  affiliation = "Department of Statistics, Stanford University, Stanford, CA
                 94305, USA. Department of Statistics, Stanford University,
                 Stanford, CA 94305, USA. Department of Statistics, Stanford
                 University, Stanford, CA 94305, USA.",
  abstract    = "We study the variability of predictions made by bagged
                 learners and random forests, and show how to estimate standard
                 errors for these methods. Our work builds on variance
                 estimates for bagging proposed by Efron (1992, 2013) that are
                 based on the jackknife and the infinitesimal jackknife (IJ).
                 In practice, bagged predictors are computed using a finite
                 number B of bootstrap replicates, and working with a large B
                 can be computationally expensive. Direct applications of
                 jackknife and IJ estimators to bagging require B = $\Theta$(n
                 (1.5)) bootstrap replicates to converge, where n is the size
                 of the training set. We propose improved versions that only
                 require B = $\Theta$(n) replicates. Moreover, we show that the
                 IJ estimator requires 1.7 times less bootstrap replicates than
                 the jackknife to achieve a given accuracy. Finally, we study
                 the sampling distributions of the jackknife and IJ variance
                 estimates themselves. We illustrate our findings with multiple
                 experiments and simulation studies.",
  journal     = "J. Mach. Learn. Res.",
  volume      =  15,
  number      =  1,
  pages       = "1625--1651",
  month       =  jan,
  year        =  2014,
  keywords    = "Monte Carlo noise; bagging; jackknife methods; variance
                 estimation"}
                 """),
         description=("Confidence Intervals for Random Forests:",
                      "The Jackknife and the Infinitesimal Jackknife"),
         path='sklforestci')


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
    n_trees = forest.n_estimators
    inbag = np.zeros((n_samples, n_trees))
    sample_idx = []
    for t_idx in range(n_trees):
        sample_idx.append(
          _generate_sample_indices(forest.estimators_[t_idx].random_state,
                                   n_samples))
        inbag[:, t_idx] = np.bincount(sample_idx[-1], minlength=n_samples)
    return inbag


def _core_computation(X_train, X_test, inbag, pred_centered, n_trees):
    cov_hat = np.zeros((X_train.shape[0], X_test.shape[0]))

    for t_idx in range(n_trees):
        inbag_r = (inbag[:, t_idx] - 1).reshape(-1, 1)
        pred_c_r = pred_centered.T[t_idx].reshape(1, -1)
        cov_hat += np.dot(inbag_r, pred_c_r) / n_trees
    V_IJ = np.sum(cov_hat ** 2, 0)
    return V_IJ


def _bias_correction(V_IJ, inbag, pred_centered, n_trees):
    n_train_samples = inbag.shape[0]
    n_var = np.mean(np.square(inbag[0:n_trees]).mean(axis=1).T.view() -
                    np.square(inbag[0:n_trees].mean(axis=1)).T.view())
    boot_var = np.square(pred_centered).sum(axis=1) / n_trees
    bias_correction = n_train_samples * n_var * boot_var/n_trees
    V_IJ_unbiased = V_IJ - bias_correction
    return V_IJ_unbiased


def random_forest_error(forest, inbag, X_train, X_test):
    """
    Calculates error bars from scikit-learn RandomForest estimators.

    RandomForest is a regressor or classifier object
    this variance can be used to plot error bars for RandomForest objects

    Parameters
    ----------
    forest : RandomForest
        Regressor or Classifier object.

    inbag : ndarray
        The inbag matrix that fit the data.

    X : ndarray
        An array with shape (n_sample, n_features).

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
    pred = np.array([tree.predict(X_test) for tree in forest]).T
    pred_mean = np.mean(pred, 0)
    pred_centered = pred - pred_mean
    n_trees = forest.n_estimators
    V_IJ = _core_computation(X_train, X_test, inbag, pred_centered, n_trees)
    V_IJ_unbiased = _bias_correction(V_IJ, inbag, pred_centered, n_trees)
    return V_IJ_unbiased
