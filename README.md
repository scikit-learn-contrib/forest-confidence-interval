# `forestci`: confidence intervals for Forest algorithms

[![Travis Status](https://travis-ci.org/scikit-learn-contrib/forest-confidence-interval.svg?branch=master)](https://travis-ci.org/scikit-learn-contrib/forest-confidence-interval)
[![Coveralls Status](https://coveralls.io/repos/scikit-learn-contrib/forest-confidence-interval/badge.svg?branch=master&service=github)](https://coveralls.io/r/scikit-learn-contrib/forest-confidence-interval)
[![CircleCI Status](https://circleci.com/gh/scikit-learn-contrib/forest-confidence-interval.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/scikit-learn-contrib/forest-confidence-interval/tree/master)
[![status](http://joss.theoj.org/papers/b40f03cc069b43b341a92bd26b660f35/status.svg)](http://joss.theoj.org/papers/b40f03cc069b43b341a92bd26b660f35)

Forest algorithms are powerful [ensemble methods](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble) for classification and regression. 
However, predictions from these algorithms do contain some amount of error. 
Prediction variability can illustrate how influential
the training set is for producing the observed random forest predictions.

`forest-confidence-interval` is a Python module that adds a calculation of
variance and computes confidence intervals to the basic functionality
implemented in scikit-learn random forest regression or classification objects.
The core functions calculate an in-bag and error bars for random forest
objects.

This module is based on R code from Stefan Wager 
([`randomForestCI`](https://github.com/swager/randomForestCI) deprecated in favor of [`grf`](https://github.com/swager/grf))
and is licensed under the MIT open source license (see [LICENSE](LICENSE)).
The present project makes the algorithm compatible with [`scikit-learn`](https://scikit-learn.org/stable/).

To get the proper confidence interval, you need to use a large number of trees (estimators). 
The [calibration routine](https://github.com/scikit-learn-contrib/forest-confidence-interval/pull/114) 
(which can be included or excluded on top of the algorithm) tries to extrapolate
the results for an infinite number of trees, but it is instable and it can cause numerical errors:
if this is the case, the suggestion is to exclude it with `calibrate=False` 
and test increasing the number of trees in the model to reach convergence.

## Installation and Usage

Before installing the module you will need `numpy`, `scipy` and `scikit-learn`.

To install `forest-confidence-interval` execute:
```
pip install forestci
```
If would like to install the development version of the software use:

```shell
pip install git+git://github.com/scikit-learn-contrib/forest-confidence-interval.git
```

Usage:

```python
import import forestci as fci
ci = fci.random_forest_error(
  forest=model, # scikit-learn Forest model fitted on X_train
  X_train_shape=X_train.shape,
  X_test=X, # the samples you want to compute the CI
  inbag=None,
  calibrate=True,
  memory_constrained=False,
  memory_limit=None,
  y_output=0 # in case of multioutput model, consider target 0
)
```

## Examples

The examples (gallery below) demonstrates the package functionality with random forest classifiers and regression models.
The regression example uses a popular UCI Machine Learning data set on cars while the classifier example simulates how to add measurements of uncertainty to tasks like predicting spam emails.

[Examples gallery](http://contrib.scikit-learn.org/forest-confidence-interval/auto_examples/index.html)

## Contributing

Contributions are very welcome, but we ask that contributors abide by the
[contributor covenant](http://contributor-covenant.org/version/1/4/).

To report issues with the software, please post to the
[issue log](https://github.com/scikit-learn-contrib/forest-confidence-interval/issues)
Bug reports are also appreciated, please add them to the issue log after
verifying that the issue does not already exist.
Comments on existing issues are also welcome.

Please submit improvements as pull requests against the repo after verifying
that the existing tests pass and any new code is well covered by unit tests.
Please write code that complies with the Python style guide,
[PEP8](https://www.python.org/dev/peps/pep-0008/).

E-mail [Ariel Rokem](mailto:arokem@gmail.com), [Kivan Polimis](mailto:kivan.polimis@gmail.com), or [Bryna Hazelton](mailto:brynah@phys.washington.edu ) if you have any questions, suggestions or feedback.

## Testing

Requires installation of `nose` package. Tests are located in the `forestci/tests` folder
and can be run with the `nosetests` command in the main directory.

## Citation

Click on the JOSS status badge for the Journal of Open Source Software article on this project.
The BibTeX citation for the JOSS article is below:

```
@article{polimisconfidence,
  title={Confidence Intervals for Random Forests in Python},
  author={Polimis, Kivan and Rokem, Ariel and Hazelton, Bryna},
  journal={Journal of Open Source Software},
  volume={2},
  number={1},
  year={2017}
}
```
