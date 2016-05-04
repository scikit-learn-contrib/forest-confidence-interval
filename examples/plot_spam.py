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
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
import sklearn.cross_validation as xval
import sklforestci as fci


def get_spam_data():
    spam_data_url = ("http://archive.ics.uci.edu/ml/machine-learning-"
                     "databases/spambase/spambase.data")
    spam_csv = "spam_data.csv"
    urlretrieve(spam_data_url, spam_csv)
    spam_names_url = ("http://archive.ics.uci.edu/ml/machine-learning-"
                      "databases/spambase/spambase.names")

    spam_names = "spam_names.txt"
    urlretrieve(spam_names_url, spam_names)
    spam_names = np.recfromcsv("spam_names.txt", skip_header=30,
                               usecols=np.arange(1))
    spam_names = spam_names['1']
    spam_names = [n.split(':')[0] for n in spam_names] + ['spam']
    spam_data = np.recfromcsv("spam_data.csv", delimiter=",",
                              names=spam_names)
    return spam_data

spam_data = get_spam_data()

spam_X = np.matrix(np.array(spam_data.tolist()))
spam_X = np.delete(spam_X, -1, 1)

spam_y = spam_data["spam"]

spam_X_train, spam_X_test, spam_y_train, spam_y_test = xval.train_test_split(
                                                       spam_X, spam_y,
                                                       test_size=0.2)

n_trees = 2000
spam_RFC = RandomForestClassifier(max_features=5, n_estimators=n_trees,
                                  random_state=42)
spam_RFC.fit(spam_X_train, spam_y_train)

spam_inbag = fci.calc_inbag(spam_X_train.shape[0], spam_RFC)

spam_V_IJ_unbiased = fci.random_forest_error(spam_RFC, spam_inbag,
                                             spam_X_train, spam_X_test)

spam_y_hat = spam_RFC.predict_proba(spam_X_test)


idx = np.where(spam_y_test == 1)[0]
plt.errorbar(spam_y_hat[idx, 1], np.sqrt(spam_V_IJ_unbiased[idx]),
             fmt='.', alpha=0.75)

idx = np.where(spam_y_test == 0)[0]
plt.errorbar(spam_y_hat[idx, 1], np.sqrt(spam_V_IJ_unbiased[idx]),
             fmt='.', alpha=0.75)
plt.show()
