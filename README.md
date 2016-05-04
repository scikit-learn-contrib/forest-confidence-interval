# `sklearn-forest-ci`: Confidence intervals for Forest algorithms

[![Travis Status](https://travis-ci.org/uwescience/sklearn-forest-ci.svg?branch=master)](https://travis-ci.org/uwescience/sklearn-forest-ci)
[![Coveralls Status](https://coveralls.io/repos/uwescience/sklearn-forest-ci/badge.svg?branch=master&service=github)](https://coveralls.io/r/uwescience/sklearn-forest-ci)
[![CircleCI Status](https://circleci.com/gh/uwescience/sklearn-forest-ci.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/uwescience/sklearn-forest-ci/tree/master)

`sklearn-forest-ci` is a Python module for calculating variance and adding
confidence intervals to scikit-learn random forest regression or classification
objects. The core functions calculate an in-bag (define) and error bars for
random forest objects

Compatible with Python2.7 and Python3.5

This module is based on R code from Stefan Wager (see important links below)
and licensed under the MIT Open Source Initiative


## Important Links
scikit-learn - http://scikit-learn.org/
Stefan Wager's `randomForestCI` - https://github.com/swager/randomForestCI

## Installation and Usage
Before installing the module you will need `numpy`, `scipy` and `scikit-learn`.
```
pip install numpy scipy scikit-learn
```

## Example
.. code-block:: python

	import sklforestci as fci

.. literalinclude:: plot_mpg.py
   :lines: 1,3,5-10,20-
