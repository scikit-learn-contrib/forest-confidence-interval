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
  affiliation: University of Washington
- name: Ariel Rokem
  orcid: 0000-0003-0679-1985
  affiliation: eScience Institute, University of Washington
- name: Bryna Hazelton
  orcid: 0000-0001-7532-645X
  affiliation: eScience Institute, University of Washington
date: 24 September 2016
bibliography: paper.bib
---

# Summary
Random forests are a method for predicting numerous ensemble learning tasks. The variability in predictions is important for measuring standard errors and estimating standard errors. Forest-Confidence-interval is a Python module for calculating variance and adding confidence intervals to scikit-learn random forest regression or classification objects. The core functions calculate an in-bag and error bars for random forest objects. This module is an implementation of an algorithm developed by @wager_confidence_2014 and previously implemented in R here [@wager_randomforestci_2016].

# Examples gallery
## Regression example
-![plot-mpg](plot_mpg.png)

## Classification example
-![plot-spam](plot_spam.png)


# References
