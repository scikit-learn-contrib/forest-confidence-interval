# `forestci`: confidence intervals for Forest algorithms

[![Python package](https://github.com/scikit-learn-contrib/forest-confidence-interval/actions/workflows/pythonpackage.yml/badge.svg?branch=master)](https://github.com/scikit-learn-contrib/forest-confidence-interval/actions/workflows/pythonpackage.yml)
[![Documentation](https://github.com/scikit-learn-contrib/forest-confidence-interval/actions/workflows/docbuild.yml/badge.svg?branch=master)](https://contrib.scikit-learn.org/forest-confidence-interval/)
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

Version 0.8 supports scikit-learn 1.0 through 1.9, including the signature
changes introduced in scikit-learn 1.9.

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

To install `forest-confidence-interval`, run:

```
python -m pip install forestci
```

To install the development version from GitHub, use:

```shell
python -m pip install git+https://github.com/scikit-learn-contrib/forest-confidence-interval.git
```

Usage:

```python
import forestci as fci
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

The examples (gallery below) demonstrate the package functionality with random forest classifiers and regression models.
The regression examples use a bundled copy of the
[UCI Auto MPG dataset](https://doi.org/10.24432/C5859H), while the classifier
example simulates how to add measurements of uncertainty to tasks like
predicting spam emails. Keeping the regression data in the repository makes
documentation builds reproducible without relying on an external data service.

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

The project uses a `pyproject.toml` build configuration and a `src/` package
layout. Runtime, test, documentation, and development dependencies are declared
as project dependency groups, while tests live in the top-level `tests/`
directory.

For a complete development environment, install the editable package and its
development dependencies:

```shell
python -m pip install -e ".[dev]"
python -m pytest
```

To test only against the scikit-learn version already installed in the current
environment, install the test runner and then install `forestci` without
resolving dependencies:

```shell
python -m pip install pytest==9.0.3
python -m pip install --no-deps -e .
python -m pytest
```

The GitHub Actions test matrix selects the newest bugfix release from every
scikit-learn minor series from 1.0 through 1.9. Scikit-learn 1.0–1.2 are tested
on Python 3.10 because those releases do not provide Python 3.12 wheels; all
newer series are tested on Python 3.12. A rolling `scikit-learn<2.0` job also
tests the latest available 1.x release.

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
