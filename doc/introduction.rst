.. _introduction:

Introduction
============
`sklforestci` calculates the variance and error bars from scikit-learn
RandomForest regressor or classifier objects. The unbiased variance for a
RandomForest object is returned in an array for plotting.

The calculation of error is based on the infinite jackknife variance, as
described in [Wager2014]_ and is a Python implementation of the R code
provided at: https://github.com/swager/randomForestCI

.. [Wager2014] S. Wager, T. Hastie, B. Efron. "Confidence Intervals for
       Random Forests: The Jackknife and the Infinitesimal Jackknife", Journal
       of Machine Learning Research vol. 15, pp. 1625-1651, 2014.
