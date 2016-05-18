"""
=========================================
Plotting Classification Forest Error Bars
=========================================

Plot error bars for scikit learn RandomForest Classification objects. The
calculation of error is based on the infinitesimal jackknife variance, as
described in [Wager2014]_

.. [Wager2014] S. Wager, T. Hastie, B. Efron. "Confidence Intervals for
   Random Forests: The Jackknife and the Infinitesimal Jackknife", Journal
   of Machine Learning Research vol. 15, pp. 1625-1651, 2014.
"""

import numpy as np
from matplotlib import pyplot as plt
import sklearn.cross_validation as xval
from sklearn.ensemble import RandomForestClassifier
import sklforestci as fci
from sklearn.datasets import make_classification

spam_X, spam_y = make_classification(5000)

# split mpg data into training and test set
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
             fmt='.', alpha=0.75)

idx = np.where(spam_y_test == 0)[0]
plt.errorbar(spam_y_hat[idx, 1], np.sqrt(spam_V_IJ_unbiased[idx]),
             fmt='.', alpha=0.75)
plt.xlabel('Prediction')
plt.ylabel('Standard Deviation Estimate')
plt.show()
