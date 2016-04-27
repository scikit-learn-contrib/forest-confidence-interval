"""
===========================
Plotting Forest Error bars
===========================

Explanation explanation explanation [Reference2001]_

.. [Reference2001] Author, A., Author, B. (2001). Title of the paper.
   Journal of important results 1: 1

"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
import sklearn.cross_validation as xval
import sklforestci as fci
sns.set()
# get_ipython().magic(u'matplotlib inline')

X = np.arange(100).reshape(100, 1)
y = np.random.randn(100)
plt.plot(X, y)
plt.show()

# Regression example
mpg_names = ["mpg", "cyl", "disp", "hp", "weight",
             "accel", "year", "origin", "name"]
mpg_df = pd.read_csv(("http://archive.ics.uci.edu/ml/machine-learning-"
                      "databases/auto-mpg/auto-mpg.data"), names=mpg_names,
                     sep="\s+")
mpg_df["hp"] = pd.to_numeric(mpg_df["hp"], errors="coerce")
mpg_df = mpg_df.dropna()
mpg_y = mpg_df["mpg"].as_matrix()
mpg_X = mpg_df.drop(["mpg", "name"], axis=1).as_matrix()
mpg_X_train, mpg_X_test, mpg_y_train, mpg_y_test = xval.train_test_split(
                                                   mpg_X, mpg_y,
                                                   test_size=0.25,
                                                   random_state=42)

n_trees = 2000


mpg_forest = RandomForestRegressor(n_estimators=n_trees)
mpg_forest.fit(mpg_X_train, mpg_y_train)
mpg_inbag = fci.calc_inbag(mpg_X_train.shape[0], mpg_forest)

mpg_V_IJ_unbiased = fci.random_forest_error(mpg_forest, mpg_inbag,
                                            mpg_X_train, mpg_X_test)

mpg_y_hat = mpg_forest.predict(mpg_X_test)

plt.errorbar(mpg_y_test, mpg_y_hat, yerr=np.sqrt(mpg_V_IJ_unbiased), fmt='o')
plt.plot([10, 45], [10, 45], '--')
plt.show()

# Classification example
spam_names = pd.read_csv(("http://archive.ics.uci.edu/ml/machine-learning-"
                          "databases/spambase/spambase.names"), skiprows=30)
spam_names = spam_names['1']
spam_names = [n.split(':')[0] for n in spam_names] + ['spam']
spam_df = pd.read_csv(("http://archive.ics.uci.edu/ml/machine-learning-"
                       "databases/spambase/spambase.data"), names=spam_names)

spam_y = spam_df["spam"].as_matrix()
spam_df["spam"].as_matrix()
spam_X = spam_df.drop(["spam"], axis=1).as_matrix()

spam_X_train, spam_X_test, spam_y_train, spam_y_test = xval.train_test_split(
                                                       spam_X, spam_y,
                                                       test_size=0.2)


spam_RFC = RandomForestClassifier(max_features=5, n_estimators=n_trees)
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
