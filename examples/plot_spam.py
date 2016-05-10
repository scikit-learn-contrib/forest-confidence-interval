"""
=========================================
Plotting Classification Forest Error Bars
=========================================

Plot error bars for scikit learn RandomForest Classification objects. The
calculation of error is based on the infinite jackknife variance, as described
in [Wager2014]_

.. [Wager2014] S. Wager, T. Hastie, B. Efron. "Confidence Intervals for
   Random Forests: The Jackknife and the Infinitesimal Jackknife", Journal
   of Machine Learning Research vol. 15, pp. 1625-1651, 2014.
"""

# Classification example
# try:
#     from urllib import urlretrieve
# except ImportError:
#     from urllib.request import urlretrieve
# import tempfile
# import os
# import os.path as op
import numpy as np
from matplotlib import pyplot as plt
# import sklearn.datasets as ds
import sklearn.cross_validation as xval
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
import sklforestci as fci


# def get_spam_data():
#     """
#     Stores .npy file that is the spam email data set from the UCI machine
#     learning database
#
#     Parameters
#     ----------
#     None
#
#     Returns
#     -------
#     Numpy ndarray with email characteristics
#
#     """
#     data_home = ds.get_data_home()
#     spam_file = op.join(data_home, "spam_data.npy")
#     if not op.exists(spam_file):
#
#         spam_data_url = ("http://archive.ics.uci.edu/ml/machine-learning-"
#                          "databases/spambase/spambase.data")
#         spam_csv_file = op.join(data_home, 'spam.csv')
#         urlretrieve(spam_data_url, spam_csv_file)
#
#         spam_names_url = ("http://archive.ics.uci.edu/ml/machine-learning-"
#                           "databases/spambase/spambase.names")
#         spam_names_file = op.join(data_home, 'spam.names')
#         urlretrieve(spam_names_url, spam_names_file)
#         spam_names = np.recfromcsv(spam_names_file, skip_header=30,
#                                    usecols=np.arange(1))
#         spam_names = spam_names['1']
#         try:  # Python 2
#             spam_names = [n.split(':')[0] for n in spam_names] + ['spam']
#         except TypeError:  # Python 3
#             spam_names = ([n.decode().split(':')[0] for n in spam_names] +
#                           ['spam'])
#
#         spam_data = np.recfromcsv(spam_csv_file, delimiter=",",
#                                   names=spam_names)
#         np.save(spam_file, spam_data)
#         os.remove(spam_csv_file)
#         os.remove(spam_names_file)
#
#     spam_data = np.load(spam_file)
#     return spam_data
#
# # retreive spam email data from machine learning library
# spam_data = get_spam_data()

# # separate spam email data into predictors and outcome variable
# spam_X = np.matrix(np.array(spam_data.tolist()))
# spam_X = np.delete(spam_X, -1, 1)
# spam_y = spam_data["spam"]

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
