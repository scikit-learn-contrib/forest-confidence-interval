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
date: 27 March 2017
bibliography: paper.bib
---

# Summary
Random forests are a method for predicting numerous ensemble learning tasks. The variability in predictions is important for measuring standard errors and estimating standard errors. Forest-Confidence-Interval is a Python module for calculating variance and adding confidence intervals to `scikit-learn` random forest regression or classification objects. The core functions calculate an in-bag and error bars for random forest objects. Our software is designed for individuals using `scikit-learn` random  forest objects that want to add estimates of uncertainty to random forest predictors. This module is an implementation of an algorithm developed by @wager_confidence_2014 and previously implemented in R here [@wager_randomforestci_2016].


# Examples gallery

This gallery uses two standard machine learning examples to graphically show the package's calculation of standard errors. The example data was imported as a data frame and separated into X (predictors) and y matrices (predictions). Then, `scikit-learn` functions split the example data  into training and  test data for creating random forest objects. After creating the random forest objects, the package's `random_forest_error` function used the random forest object, inbag, train, and test data to create variance estimates. 


## Regression example

### No variance estimated
-![plot-mpg-no-variance](plot_mpg_no_variance.png)

### Plot with variance
-![plot-mpg-variance](plot_mpg.png)

## Classification example
-![plot-spam](plot_spam.png)


# References
