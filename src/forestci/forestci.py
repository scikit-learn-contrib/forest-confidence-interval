"""
Forest confidence intervals.

Calculate confidence intervals for scikit-learn RandomForestRegressor and
RandomForestClassifier predictions.
"""

import copy
import warnings
from numbers import Integral

import numpy as np

import sklearn
from sklearn.base import is_classifier
from sklearn.ensemble._forest import BaseForest
from sklearn.ensemble._forest import (_generate_sample_indices,
                                      _get_n_samples_bootstrap)
from sklearn.ensemble._bagging import BaseBagging
from sklearn.utils.fixes import parse_version

from .calibration import calibrateEB

# scikit-learn 1.9 made `sample_weight` a required argument of the private
# helpers `_get_n_samples_bootstrap` and `_generate_sample_indices`
# (https://github.com/scikit-learn/scikit-learn/pull/31529). Passing None
# reproduces the previous uniform-bootstrap behavior. Compare against
# "1.9.0.dev0" so 1.9 pre-releases (which already carry the new signature and
# sort before the final under PEP 440) are also covered.
_SKLEARN_GE_19 = (
    parse_version(sklearn.__version__) >= parse_version("1.9.0.dev0")
)

__all__ = ("calc_inbag", "random_forest_error", "_bias_correction",
           "_core_computation")


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
        e_s = "Cannot calculate the inbag from a forest that has bootstrap=False"
        raise ValueError(e_s)

    n_trees = forest.n_estimators
    inbag = np.zeros((n_samples, n_trees))
    sample_idx = []
    if isinstance(forest, BaseForest):
        if _SKLEARN_GE_19:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples, forest.max_samples, None
            )
        else:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples, forest.max_samples
            )

        for t_idx in range(n_trees):
            random_state = forest.estimators_[t_idx].random_state
            if _SKLEARN_GE_19:
                sample_idx.append(
                    _generate_sample_indices(
                        random_state,
                        n_samples,
                        n_samples_bootstrap,
                        None,
                    )
                )
            else:
                sample_idx.append(
                    _generate_sample_indices(
                        random_state,
                        n_samples,
                        n_samples_bootstrap,
                    )
                )
            inbag[:, t_idx] = np.bincount(sample_idx[-1], minlength=n_samples)
    elif isinstance(forest, BaseBagging):
        for t_idx, estimator_sample in enumerate(forest.estimators_samples_):
            sample_idx.append(estimator_sample)
            inbag[:, t_idx] = np.bincount(sample_idx[-1], minlength=n_samples)

    return inbag


def _core_computation(
    X_train_shape,
    X_test,
    inbag,
    pred_centered,
    n_trees,
    memory_constrained=False,
    memory_limit=None,
    test_mode=False,
    show_progress=False,
):
    """
    Helper function, that performs the core computation

    Parameters
    ----------
    X_train_shape : tuple (int, int)
        Shape (n_train_sample, n_features).

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

    show_progress: boolean, optional
        Whether to display progress while processing memory-constrained chunks.
        Requires the optional ``tqdm`` dependency. Default: False.
    """
    if not memory_constrained:
        return np.sum((np.dot(inbag - 1, pred_centered.T) / n_trees) ** 2, 0)

    if not memory_limit:
        raise ValueError("If memory_constrained=True, must provide", "memory_limit.")

    # Assumes double precision float
    chunk_size = int((memory_limit * 1e6) / (8.0 * X_train_shape[0]))

    if chunk_size == 0:
        min_limit = 8.0 * X_train_shape[0] / 1e6
        raise ValueError(
            "memory_limit provided is too small."
            + "For these dimensions, memory_limit must "
            + "be greater than or equal to %.3e" % min_limit
        )

    # Predictions already normalize a single sample to one row.
    n_test_samples = pred_centered.shape[0]
    chunk_edges = np.arange(0, n_test_samples + chunk_size, chunk_size)
    inds = range(n_test_samples)
    chunks = [
        inds[chunk_edges[i] : chunk_edges[i + 1]] for i in range(len(chunk_edges) - 1)
    ]
    if test_mode:
        print("Number of chunks: %d" % (len(chunks),))

    if show_progress:
        try:
            from tqdm.auto import tqdm

            description = f"Computing V_IJ ({n_trees} trees)"
            chunks_iter = tqdm(chunks, desc=description)
        except ImportError:
            warnings.warn(
                "tqdm must be installed to use the progress indicator. "
                "Install forestci[progress] or set show_progress=False.",
                stacklevel=2,
            )
            chunks_iter = chunks
    else:
        chunks_iter = chunks

    V_IJ = np.concatenate(
        [
            np.sum((np.dot(inbag - 1, pred_centered[chunk].T) / n_trees) ** 2, 0)
            for chunk in chunks_iter
        ]
    )
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
    n_var = np.mean(
        np.square(inbag).mean(axis=1).T.view()
        - np.square(inbag.mean(axis=1)).T.view()
    )
    boot_var = np.square(pred_centered).sum(axis=1) / n_trees
    bias_correction = n_train_samples * n_var * boot_var / n_trees
    V_IJ_unbiased = V_IJ - bias_correction
    return V_IJ_unbiased


def _centered_prediction_forest(forest, X_test, y_output=None, class_index=None):
    """
    Center the tree predictions by the mean prediction (forest)

    The centering is done for all provided test samples.
    This function allows unit testing for internal correctness.

    Parameters
    ----------
    forest : RandomForest
        Regressor or Classifier object.

    X_test : ndarray
        An array with shape (n_test_sample, n_features). The design matrix
        for testing data

    Returns
    -------
    pred_centered : ndarray
        An array with shape (n_test_sample, n_estimators).
        The predictions of each single tree centered by the
        mean prediction (i.e. the prediction of the forest)

    """
    # In case the user provided a (n_features)-shaped array for a single sample
    #  shape it as (1, n_features)
    # NOTE: a single-feature set of samples needs to be provided with shape
    #       (n_samples, 1) or it will be wrongly interpreted!
    if len(X_test.shape) == 1:
        X_test = X_test.reshape(1, -1)

    if class_index is None:
        pred = np.array([tree.predict(X_test) for tree in forest])
    else:
        # Binary probabilities are complements. Center one shared coordinate
        # so their variances (and inputs to the sensitive EB fit) are identical
        # up to sign, rather than differing through subtraction roundoff.
        probability_index = 1 if len(forest.classes_) == 2 else class_index
        pred = np.array([
            tree.predict_proba(X_test)[:, probability_index] for tree in forest
        ])
    if 'n_outputs_' in dir(forest) and forest.n_outputs_ > 1:
        pred = pred[:,:,y_output]

    pred_mean = np.mean(pred, 0)

    centered = (pred - pred_mean).T
    if class_index == 0 and len(forest.classes_) == 2:
        centered = -centered
    return centered


def random_forest_error(
    forest,
    X_train_shape,
    X_test,
    inbag=None,
    calibrate=True,
    memory_constrained=False,
    memory_limit=None,
    y_output=None,
    show_progress=False,
    *,
    class_index=None,
):
    """
    Calculate error bars from scikit-learn RandomForest estimators.

    RandomForest is a regressor or classifier object
    this variance can be used to plot error bars for RandomForest objects

    Parameters
    ----------
    forest : RandomForest
        Regressor or Classifier object.

    X_train_shape : tuple (int, int)
        Shape (n_train_sample, n_features) of the design matrix for training data.

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

    y_output: int, mandatory only for MultiOutput regressor.
        In case of MultiOutput regressor, indicate the index of the target to
        analyse. The program will return the IJ variance related to that target
        only.

    show_progress: boolean, optional
        Whether to display progress while processing memory-constrained chunks.
        This has no effect unless ``memory_constrained=True`` and requires the
        optional ``tqdm`` dependency. Calibration displays a second bar for its
        reduced forest computation. Default: False.

    class_index : int, optional, keyword-only
        Column index in ``forest.classes_`` of the class probability whose
        sampling variance is requested. Supported for single-output forest
        classifiers. This is a column position, not a class label. With this
        argument, individual trees' ``predict_proba`` values are used.
        Without it, the legacy hard-prediction calculation is preserved:
        binary classifiers estimate vote-fraction variance; multiclass
        classifiers emit a warning because numeric class codes have no
        meaningful variance. Regressors must leave this argument unset.

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

    Class-probability results are marginal sampling variances of the fitted
    probabilities, not outcome variances or simultaneous confidence regions.
    Calibration mitigates finite-tree noise in the variance estimate; it does
    not calibrate class probabilities or remove prediction bias.

    .. [Wager2014] S. Wager, T. Hastie, B. Efron. "Confidence Intervals for
       Random Forests: The Jackknife and the Infinitesimal Jackknife", Journal
       of Machine Learning Research vol. 15, pp. 1625-1651, 2014.
    """

    if is_classifier(forest):
        multioutput_classifier = getattr(forest, "n_outputs_", 1) > 1
        if class_index is not None and multioutput_classifier:
            raise ValueError("Multi-output classifiers are not supported.")
        if class_index is not None:
            if not isinstance(forest, BaseForest):
                raise ValueError("class_index requires a forest classifier.")
            if (isinstance(class_index, (bool, np.bool_))
                    or not isinstance(class_index, Integral)
                    or not 0 <= class_index < len(forest.classes_)):
                raise ValueError(
                    "class_index must be an integer column index in forest.classes_."
                )
        elif not multioutput_classifier and len(forest.classes_) > 2:
            warnings.warn(
                "Multiclass hard-prediction variance depends on arbitrary class "
                "codes. Specify class_index to estimate class-probability "
                "variance. The legacy multiclass calculation is deprecated.",
                FutureWarning,
                stacklevel=2,
            )
    elif class_index is not None:
        raise ValueError("class_index is only supported for forest classifiers.")

    if 'n_outputs_' in dir(forest) and forest.n_outputs_ > 1 and y_output == None:
        e_s = "MultiOutput regressor: specify the index of the target to analyse (y_output)"
        raise ValueError(e_s)

    if inbag is None:
        inbag = calc_inbag(X_train_shape[0], forest)

    pred_centered = _centered_prediction_forest(
        forest, X_test, y_output, class_index=class_index
    )
    n_trees = forest.n_estimators
    V_IJ = _core_computation(
        X_train_shape,
        X_test,
        inbag,
        pred_centered,
        n_trees,
        memory_constrained,
        memory_limit,
        show_progress=show_progress,
    )
    V_IJ_unbiased = _bias_correction(V_IJ, inbag, pred_centered, n_trees)

    # Correct for cases where resampling is done without replacement:
    if np.max(inbag) == 1:
        variance_inflation = 1 / (1 - np.mean(inbag)) ** 2
        V_IJ_unbiased *= variance_inflation

    if not calibrate:
        return V_IJ_unbiased

    if V_IJ_unbiased.shape[0] <= 20:
        print("No calibration with n_samples <= 20: ", 
                 "consider using more n_estimators in your model, ", 
                 "for more accurate ci and to avoid negative values.")
        return V_IJ_unbiased
    if calibrate:
        # Calibration is a correction for converging quicker to the case of infinite n_estimators,
        # as presented in Wager (2014) http://jmlr.org/papers/v15/wager14a.html
        calibration_ratio = 2
        n_sample = np.ceil(n_trees / calibration_ratio)
        new_forest = copy.deepcopy(forest)
        random_idx = np.random.permutation(len(new_forest.estimators_))[: int(n_sample)]
        new_forest.estimators_ = list(np.array(new_forest.estimators_)[random_idx])
        if hasattr(new_forest, "_seeds"):
            new_forest._seeds = new_forest._seeds[random_idx]

        new_forest.n_estimators = int(n_sample)

        results_ss = random_forest_error(
            new_forest,
            X_train_shape,
            X_test,
            inbag=inbag[:, random_idx],
            calibrate=False,
            memory_constrained=memory_constrained,
            memory_limit=memory_limit,
            y_output=y_output,
            show_progress=show_progress,
            class_index=class_index,
        )
        # Use this second set of variance estimates
        # to estimate scale of Monte Carlo noise
        sigma2_ss = np.mean((results_ss - V_IJ_unbiased) ** 2)
        delta = n_sample / n_trees
        sigma2 = (delta ** 2 + (1 - delta) ** 2) / (2 * (1 - delta) ** 2) * sigma2_ss

        # Use Monte Carlo noise scale estimate for empirical Bayes calibration
        V_IJ_calibrated = calibrateEB(V_IJ_unbiased, sigma2)

        return V_IJ_calibrated
