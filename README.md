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
``` python
import sklforestci as fci
from sklearn.ensemble import RandomForestRegressor
import sklearn.cross_validation as xval
from sklearn.datasets.mldata import fetch_mldata
import sklforestci as fci

mpg_data = fetch_mldata('mpg')

mpg_X = mpg_data["data"]
mpg_y = mpg_data["target"]

mpg_X_train, mpg_X_test, mpg_y_train, mpg_y_test = xval.train_test_split(
                                                   mpg_X, mpg_y,
                                                   test_size=0.25,
                                                   random_state=42
                                                   )

n_trees = 2000
mpg_forest = RandomForestRegressor(n_estimators=n_trees, random_state=42)
mpg_forest.fit(mpg_X_train, mpg_y_train)
mpg_inbag = fci.calc_inbag(mpg_X_train.shape[0], mpg_forest)

mpg_V_IJ_unbiased = fci.random_forest_error(mpg_forest, mpg_inbag,
                                            mpg_X_train, mpg_X_test)
```
