.. _general_examples:

Examples
=========

The examples use standard machine learning datasets to demonstrate how
`forestci` can be used to calculate error bars on
:class:`RandomForestRegressor` and :class:`RandomForestClassifier` objects. The
regression examples use a bundled copy of the `Auto MPG dataset
<https://doi.org/10.24432/C5859H>`_ from the UC Irvine Machine Learning
Repository, with features of different cars and their MPG. The classification
example generates synthetic data to simulate a task like that of a spam
filter: classifying items into one of two categories (e.g., spam/non-spam)
based on a number of features. The dataset attribution and license are
documented in ``data/README.md``.
