---
title: 'Confidence Intervals for Random Forests in Python'
tags:
  - Python
  - scikit-learn
  - random forest 
  - confidence intervals
authors:
- name: Kivan Polimis
  orcid: 0000-0002-3498-0479
  affiliation: eScience Institute, University of Washington
- name: Ariel Rokem
  orcid: 0000-0003-0679-1985
  affiliation: eScience Institute, University of Washington
- name: Bryna Hazelton
  orcid: 0000-0001-7532-645X
  affiliation: eScience Institute, University of Washington
date: 7 July 2017
bibliography: paper.bib
---

# Summary
Random forests are a method for predicting numerous ensemble learning tasks. The variability in predictions is important for measuring standard errors and estimating standard errors. `forest-confidence-interval` is a Python module for calculating variance and adding confidence intervals to `scikit-learn` random forest regression or classification objects. The core functions calculate an in-bag and error bars for random forest objects. Our software is designed for individuals using `scikit-learn` random forest objects that want to add estimates of uncertainty to random forest predictors. This module is an implementation of an algorithm developed by @wager_confidence_2014 and previously implemented in R here [@wager_randomforestci_2016].


# Examples gallery
The regression example uses a data-set from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Auto+MPG) with features of different cars and their MPG. The classification example generates synthetic data to simulate a task like that of a spam filter: classifying items into one of two categories (e.g., spam/non-spam) based on a number of features. The example data was imported as a data frame and separated into X (predictors) and y matrices (predictions). Then, `scikit-learn` functions split the example data into training and test data and generate a random forest object (regression or classifier). Our package's `random_forest_error` and `calc_inbag` functions use the random forest object (including training and test data) to create variance estimates that can be plotted (e.g. as confidence intervals or standard deviations). 


## Regression example

### No variance estimated
-![plot-mpg-no-variance](plot_mpg_no_variance.png)

### Plot with variance
-![plot-mpg-variance](plot_mpg.png)

## Classification example
### No variance estimated
-![plot-spam-no-variance](plot_spam_no_variance.png)

### Plot with variance
-![plot-spam](plot_spam.png)


## Community guidelines

Contributions to address any of the issues in the [issue log](https://github.com/scikit-learn-contrib/forest-confidence-interval/issues) are very welcome. Please submit improvements as pull requests against the repo after verifying that the existing tests pass and any new code is well covered by unit tests.

Bug reports are also appreciated, please add them to the issue log after verifying that the issue does not already exist. Comments on existing issues are also welcome.

Please e-mail [Ariel Rokem](mailto:arokem@gmail.com), [Kivan Polimis](mailto:kivan.polimis@gmail.com), or [Bryna Hazelton](brynah@phys.washington.edu ) if you have any questions, suggestions or feedback.



# References
