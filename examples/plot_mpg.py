"""
======================================
Plotting Regresssion Forest Error Bars
======================================

Plot error bars for scikit learn RandomForest Regression objects. The
calculation of error is based on the infinite jackknife variance, as described
in [Wager2014]_

.. [Wager2014] S. Wager, T. Hastie, B. Efron. "Confidence Intervals for
   Random Forests: The Jackknife and the Infinitesimal Jackknife", Journal
   of Machine Learning Research vol. 15, pp. 1625-1651, 2014.

"""
# Regression Forest Example
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
import sklearn.cross_validation as xval
from sklearn.datasets.mldata import fetch_mldata
import sklforestci as fci

mpg_data = fetch_mldata('mpg')

mpg_X = mpg_data["data"]
mpg_y = mpg_data["target"]

mpg_X_train, mpg_X_test, mpg_y_train, mpg_y_test = xval.train_test_split(
                                                   mpg_X, mpg_y,
                                                   test_size=0.25,
                                                   random_state=42
                                                   )

n_trees = 2000
mpg_forest = RandomForestRegressor(n_estimators=n_trees, random_state=42)
mpg_forest.fit(mpg_X_train, mpg_y_train)
mpg_inbag = fci.calc_inbag(mpg_X_train.shape[0], mpg_forest)

mpg_V_IJ_unbiased = fci.random_forest_error(mpg_forest, mpg_inbag,
                                            mpg_X_train, mpg_X_test)

mpg_y_hat = mpg_forest.predict(mpg_X_test)

# Plot forest prediction for MPG and reported MPG
plt.errorbar(mpg_y_test, mpg_y_hat, yerr=np.sqrt(mpg_V_IJ_unbiased), fmt='o')
plt.plot([5, 45], [5, 45], '--')
plt.xlabel('Reported MPG')
plt.ylabel('Predicted MPG')
plt.show()
