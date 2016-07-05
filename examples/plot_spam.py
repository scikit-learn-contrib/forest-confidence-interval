"""
=========================================
Plotting Classification Forest Error Bars
=========================================

This example demonstrates the calculation of confidence intervals for
:class:`sklearn.ensemble.RandomForestClassifier` objects.

The data used here are synthetically generated to simulate a data-set in which
email messages are labeled as spam based on 20 different features (the default
of :func:`sklearn.datasets.make_classification`).
"""

import numpy as np
from matplotlib import pyplot as plt
import sklearn.cross_validation as xval
from sklearn.ensemble import RandomForestClassifier
import sklforestci as fci
from sklearn.datasets import make_classification

spam_X, spam_y = make_classification(5000)

# split the datainto training and test set
spam_X_train, spam_X_test, spam_y_train, spam_y_test = xval.train_test_split(
                                                       spam_X, spam_y,
                                                       test_size=0.2)

# create RandomForestClassifier
n_trees = 500
spam_RFC = RandomForestClassifier(max_features=5, n_estimators=n_trees,
                                  random_state=42)
spam_RFC.fit(spam_X_train, spam_y_train)
spam_y_hat = spam_RFC.predict_proba(spam_X_test)

# calculate inbag and unbiased variance
spam_inbag = fci.calc_inbag(spam_X_train.shape[0], spam_RFC)
spam_V_IJ_unbiased = fci.random_forest_error(spam_RFC, spam_inbag,
                                             spam_X_train, spam_X_test)

# Plot forest prediction for emails and standard deviation for estimates
# Blue points are spam emails; Green points are non-spam emails
idx = np.where(spam_y_test == 1)[0]
plt.errorbar(spam_y_hat[idx, 1], np.sqrt(spam_V_IJ_unbiased[idx]),
             fmt='.', alpha=0.75, label='spam')

idx = np.where(spam_y_test == 0)[0]
plt.errorbar(spam_y_hat[idx, 1], np.sqrt(spam_V_IJ_unbiased[idx]),
             fmt='.', alpha=0.75, label='not spam')

plt.xlabel('Prediction (spam probability)')
plt.ylabel('Standard deviation')
plt.legend()
plt.show()
