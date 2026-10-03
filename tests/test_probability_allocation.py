"""Selected-class predictions must not retain every full probability matrix."""

import gc
import weakref

import numpy as np
from numpy.testing import assert_allclose
import pytest

from forestci.forestci import _centered_prediction_forest


@pytest.mark.parametrize(
    "n_classes, class_index", [(3, 0), (3, 2), (20, 0), (20, 19)]
)
def test_unselected_probability_columns_are_released(n_classes, class_index):
    references = []
    retained_counts = []

    class Tree:
        def __init__(self, index):
            self.index = index

        def predict_proba(self, X):
            gc.collect()
            retained_counts.append(
                sum(ref() is not None for ref in references)
            )
            probabilities = np.full((len(X), n_classes), 1.0 / n_classes)
            probabilities[:, class_index] += self.index * 0.001
            other_index = (class_index + 1) % n_classes
            probabilities[:, other_index] -= self.index * 0.001
            references.append(weakref.ref(probabilities))
            return probabilities

    class Forest(list):
        classes_ = np.arange(n_classes)
        n_outputs_ = 1

    centered = _centered_prediction_forest(
        Forest(Tree(index) for index in range(5)),
        np.zeros((20, 1)),
        class_index=class_index,
    )
    assert_allclose(
        centered, np.tile(np.arange(-2, 3) * 0.001, (20, 1)), atol=1e-16
    )
    # A loop may hold its latest result, but not every previous matrix.
    assert max(retained_counts) <= 1
    gc.collect()
    assert all(ref() is None for ref in references)
