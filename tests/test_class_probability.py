"""Class probability IJ estimates and preservation of the legacy API."""

import copy
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_classification
from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                              RandomForestClassifier, RandomForestRegressor)

from forestci import calc_inbag, random_forest_error
from forestci.forestci import _centered_prediction_forest


@pytest.fixture
def data():
    return make_classification(n_samples=100, n_features=5, n_informative=3,
                               n_redundant=0, n_classes=3, random_state=42)


def direct_variance(forest, X, inbag, k):
    predictions = np.array([t.predict_proba(X)[:, k] for t in forest]).T
    centered = predictions - predictions.mean(axis=1, keepdims=True)
    covariance = (inbag - inbag.mean(axis=1, keepdims=True)) @ centered.T
    covariance /= len(forest)
    correction = (np.var(inbag, axis=1).sum()
                  * np.mean(centered ** 2, axis=1) / len(forest))
    return (covariance ** 2).sum(axis=0) - correction


@pytest.mark.parametrize('estimator', [RandomForestClassifier, ExtraTreesClassifier])
@pytest.mark.parametrize('labels', ['numeric', 'strings'])
@pytest.mark.parametrize('memory_constrained', [False, True])
def test_probability_variance(data, estimator, labels, memory_constrained):
    X, y = data
    y = np.array(['apple', 'pear', 'plum'] if labels == 'strings'
                 else [10, 20, 100])[y]
    forest = estimator(n_estimators=40, min_samples_leaf=5, bootstrap=True,
                       random_state=4).fit(X, y)
    inbag = calc_inbag(len(X), forest)
    for k in range(3):
        result = random_forest_error(
            forest, X.shape, X[:9], class_index=np.int64(k),
            calibrate=False, memory_constrained=memory_constrained,
            memory_limit=0.001)
        assert result.shape == (9,)
        assert_allclose(result, direct_variance(forest, X[:9], inbag, k),
                        atol=1e-14)
        for sample in [X[0], X[:1]]:
            single = random_forest_error(
                forest, X.shape, sample, class_index=k, calibrate=False,
                memory_constrained=memory_constrained, memory_limit=0.001)
            assert single.shape == (1,)
            assert_allclose(single, result[:1], atol=1e-14)


def test_binary_probabilities_and_legacy_votes(data):
    X, y = data
    forest = RandomForestClassifier(n_estimators=40, min_samples_leaf=10,
                                    random_state=4).fit(X, y == 0)
    centered_votes = _centered_prediction_forest(forest, X)
    votes = np.array([t.predict(X) for t in forest]).T
    assert_allclose(centered_votes, votes - votes.mean(axis=1, keepdims=True))
    probabilities = _centered_prediction_forest(forest, X, class_index=1)
    assert not np.allclose(centered_votes, probabilities)
    legacy = random_forest_error(forest, X.shape, X, calibrate=False)
    p1 = random_forest_error(forest, X.shape, X, class_index=1, calibrate=False)
    assert not np.allclose(legacy, p1)
    for calibrate in [False, True]:
        with patch('forestci.forestci.np.random.permutation',
                   return_value=np.arange(len(forest))):
            v0 = random_forest_error(forest, X.shape, X, class_index=0,
                                    calibrate=calibrate)
            v1 = random_forest_error(forest, X.shape, X, class_index=1,
                                    calibrate=calibrate)
        assert_allclose(v0, v1, atol=1e-9, rtol=1e-5)


@pytest.mark.parametrize('memory_constrained', [False, True])
def test_calibration_keeps_class_and_inbag(data, memory_constrained):
    X, y = data
    forest = RandomForestClassifier(n_estimators=12, min_samples_leaf=4,
                                    random_state=4).fit(X, y)
    inbag = calc_inbag(len(X), forest)
    selected = np.array([9, 2, 7, 1, 11, 0])
    reduced = copy.deepcopy(forest)
    reduced.estimators_ = [forest.estimators_[i] for i in selected]
    reduced.n_estimators = len(selected)
    expected = direct_variance(forest, X, inbag, 2)
    expected_ss = direct_variance(reduced, X, inbag[:, selected], 2)
    with patch('forestci.forestci.np.random.permutation', return_value=selected), \
         patch('forestci.forestci.calc_inbag', side_effect=AssertionError), \
         patch('forestci.forestci.calibrateEB', side_effect=lambda v, s: v) as eb:
        result = random_forest_error(
            forest, X.shape, X, inbag=inbag, class_index=2,
            memory_constrained=memory_constrained, memory_limit=0.001)
    assert_allclose(result, expected, atol=1e-14)
    assert_allclose(eb.call_args.args[1], np.mean((expected_ss - expected) ** 2))


@pytest.mark.parametrize('index', [-1, 3, 1.5, '1', True, np.bool_(False)])
def test_invalid_class_index(data, index):
    X, y = data
    forest = RandomForestClassifier(n_estimators=3, random_state=4).fit(X, y)
    with pytest.raises(ValueError, match='class_index'):
        random_forest_error(forest, X.shape, X, class_index=index)


def test_unsupported_estimators_and_multiclass_warning(data):
    X, y = data
    regressor = RandomForestRegressor(n_estimators=3).fit(X, y)
    with pytest.raises(ValueError, match='class_index'):
        random_forest_error(regressor, X.shape, X, class_index=0)
    bagger = BaggingClassifier(n_estimators=3).fit(X, y)
    with pytest.raises(ValueError, match='forest classifier'):
        random_forest_error(bagger, X.shape, X, class_index=0)
    multi = RandomForestClassifier(n_estimators=3).fit(X, np.column_stack([y, y]))
    with pytest.raises(ValueError, match='Multi-output classifiers'):
        random_forest_error(multi, X.shape, X, class_index=0)
    forest = RandomForestClassifier(n_estimators=3, random_state=4).fit(X, y)
    with pytest.warns(FutureWarning, match='class_index'):
        result = random_forest_error(forest, X.shape, X, calibrate=False)
    assert result.shape == (len(X),)


def test_relabeling_and_probability_column_permutation(data):
    X, y = data
    forest = RandomForestClassifier(n_estimators=20, random_state=4).fit(X, y)
    original = [random_forest_error(forest, X.shape, X, class_index=k,
                                   calibrate=False) for k in range(3)]
    permutation = [2, 0, 1]
    # Permute the fitted probability coordinates without changing the trees.
    # Refitting after reordering labels can change split/tie decisions.
    permuted = copy.deepcopy(forest)
    permuted.classes_ = forest.classes_[permutation]
    for tree, source in zip(permuted, forest):
        tree.predict_proba = lambda X, source=source: source.predict_proba(X)[:, permutation]
    for k, source_k in enumerate(permutation):
        result = random_forest_error(permuted, X.shape, X, class_index=k,
                                     calibrate=False)
        assert_allclose(result, original[source_k])


def test_single_class_probability_is_constant(data):
    X, y = data
    forest = RandomForestClassifier(n_estimators=10, random_state=4).fit(X, y * 0)
    for calibrate in [False, True]:
        result = random_forest_error(forest, X.shape, X, class_index=0,
                                     calibrate=calibrate)
        assert_allclose(result, 0)


def test_legacy_multioutput_classifier_is_preserved(data):
    X, y = data
    forest = RandomForestClassifier(n_estimators=20, random_state=4).fit(
        X, np.column_stack([y == 0, y == 1]))
    result = random_forest_error(forest, X.shape, X, y_output=1, calibrate=False)
    assert result.shape == (len(X),)
    with pytest.raises(ValueError, match='Multi-output classifiers'):
        random_forest_error(forest, X.shape, X, y_output=1, class_index=0)
