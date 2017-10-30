# `forest-confidence-interval`: Confidence intervals for Forest algorithms

[![Travis Status](https://travis-ci.org/scikit-learn-contrib/forest-confidence-interval.svg?branch=master)](https://travis-ci.org/scikit-learn-contrib/forest-confidence-interval)
[![Coveralls Status](https://coveralls.io/repos/scikit-learn-contrib/forest-confidence-interval/badge.svg?branch=master&service=github)](https://coveralls.io/r/scikit-learn-contrib/forest-confidence-interval)
[![CircleCI Status](https://circleci.com/gh/scikit-learn-contrib/forest-confidence-interval.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/scikit-learn-contrib/forest-confidence-interval/tree/master)

`forest-confidence-interval` is a Python module for calculating variance and adding
confidence intervals to scikit-learn random forest regression or classification
objects. The core functions calculate an in-bag and error bars for
random forest objects. Our software is designed for individuals using `scikit-learn` random forest objects that want to add estimates of uncertainty to random forest predictors.


Compatible with Python2.7 and Python3.6

This module is based on R code from Stefan Wager (see important links below)
and is licensed under the MIT open source license (see [LICENSE](LICENSE))

## Important Links
scikit-learn - http://scikit-learn.org/

Stefan Wager's `randomForestCI` - https://github.com/swager/randomForestCI

## Installation and Usage
Before installing the module you will need `numpy`, `scipy` and `scikit-learn`.  
Dependencies associated with the previous modules may need root privileges to install
Consult the [API Reference](http://contrib.scikit-learn.org/forest-confidence-interval/reference/index.html) for documentation on core functionality
```
pip install numpy scipy scikit-learn
```
can also install dependencies with:

```
 pip install -r requirements.txt
 ```

To install `forest-confidence-interval` execute:
```
pip install forestci
```

or, if you are installing from the source code:
```shell
python setup.py install
```

## Why use `forest-confidence-interval`?
The variability in predictions is important for measuring standard errors and estimating standard errors. `forest-confidence-interval` is a Python module for calculating variance and adding confidence intervals to the popular python library `scikit-learn`. The software is compatible with both `scikit-learn` random forest regression or classification objects. Our software is designed for individuals using `scikit-learn` random forest objects that want to add estimates of uncertainty to random forest predictors.


## Examples
The examples (gallery below) demonstrates the package functionality with random forest classifiers and regression models.
The regression example uses a popular UCI Machine Learning data set on cars while the classifier example demonstrates how to add measurements of uncertainty to predictions of spam emails. 
[Examples gallery](http://contrib.scikit-learn.org/forest-confidence-interval/auto_examples/index.html)


## Testing
Requires installation of `nose` package. Tests are located in the `forestci/tests` folder and can be run with the `nosetests` command in the main directory

## Community guidelines

Contributions to address any of the issues in the [issue log](https://github.com/scikit-learn-contrib/forest-confidence-interval/issues) are very welcome. Please submit improvements as pull requests against the repo after verifying that the existing tests pass and any new code is well covered by unit tests.

Bug reports are also appreciated, please add them to the issue log after verifying that the issue does not already exist. Comments on existing issues are also welcome.

Please e-mail [Ariel Rokem](mailto:arokem@gmail.com), [Kivan Polimis](mailto:kivan.polimis@gmail.com), or [Bryna Hazelton](brynah@phys.washington.edu ) if you have any questions, suggestions or feedback.
