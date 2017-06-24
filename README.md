# `forest-confidence-interval`: Confidence intervals for Forest algorithms

[![Travis Status](https://travis-ci.org/scikit-learn-contrib/forest-confidence-interval.svg?branch=master)](https://travis-ci.org/scikit-learn-contrib/forest-confidence-interval)
[![Coveralls Status](https://coveralls.io/repos/scikit-learn-contrib/forest-confidence-interval/badge.svg?branch=master&service=github)](https://coveralls.io/r/scikit-learn-contrib/forest-confidence-interval)
[![CircleCI Status](https://circleci.com/gh/scikit-learn-contrib/forest-confidence-interval.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/scikit-learn-contrib/forest-confidence-interval/tree/master)

Forest algorithms are powerful
[ensemble methods](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble) for classification and regression. However, predictions from these
algorithms do contain some amount of error.

`forest-confidence-interval` is a Python module that adds a calculation of
variance and computes confidence intervals to the basic functionality
implemented in scikit-learn random forest regression or classification objects.
The core functions here calculate an in-bag and error bars for random forest
objects

`forest-confidence-interval` is compatible with both Python2.7 and Python3.5

This module is based on R code from Stefan Wager (see important links below)
and is licensed under the MIT open source license (see [LICENSE](LICENSE))

## Important Links
scikit-learn - http://scikit-learn.org/

Stefan Wager's `randomForestCI` - https://github.com/swager/randomForestCI

## Installation and Usage
Before installing the module you will need `numpy`, `scipy` and `scikit-learn`.
```
pip install numpy scipy scikit-learn
```

To install the module execute:
```
pip install forestci
```
or, if you are installing from the source code:
```shell
$ python setup.py install
```

## Examples
See [examples gallery](http://contrib.scikit-learn.org/forest-confidence-interval/auto_examples/index.html)
