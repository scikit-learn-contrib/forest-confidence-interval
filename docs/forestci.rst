Random Forest Confidence Intervals
==================================

This module creates an in-bag and calculates the variance of a prediction for
:class:`sklearn.ensemble.RandomForestRegressor` or
:class:`sklearn.ensemble.RandomForestClassifier` objects. This variance in the
prediction is calculated using the [Wager2014]_ infinitesimal jackknife variance
method. The variance can be used to plot error bars for RandomForest objects

See the `README <https://github.com/scikit-learn-contrib/forest-confidence-interval/blob/master/README.md>`_
for more information.


.. automodule:: forestci
   :members:

.. [Wager2014] S. Wager, T. Hastie, B. Efron. "Confidence Intervals for
   Random Forests: The Jackknife and the Infinitesimal Jackknife", Journal
   of Machine Learning Research vol. 15, pp. 1625-1651, 2014.
