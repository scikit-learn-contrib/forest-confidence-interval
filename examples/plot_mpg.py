"""
======================================
Plotting Regresssion Forest Error Bars
======================================

Explanation explanation explanation [Reference2001]_

.. [Reference2001] Author, A., Author, B. (2001). Title of the paper.
   Journal of important results 1: 1

"""
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


def get_mpg_data():
    url = ("http://archive.ics.uci.edu/ml/machine-learning-"
           "databases/auto-mpg/auto-mpg.data")
    outputfile = "auto-mpg_data.txt"
    urlretrieve(url, outputfile)
    mpg_names = ["mpg", "cyl", "disp", "hp", "weight",
                 "accel", "year", "origin", "name"]
    mpg_data = np.recfromcsv(outputfile, delimiter=None,
                             names=mpg_names,
                             dtype="f8,f8,f8,f8,f8,f8,f8,f8,S32",
                             usecols=np.arange(0, 9))
    mpg_data = mpg_data[~np.isnan(mpg_data["hp"])]
    return mpg_data

mpg_data = get_mpg_data()

mpg_X = np.array([mpg_data['cyl'], mpg_data['disp'], mpg_data['hp'],
                  mpg_data['weight'], mpg_data['accel'], mpg_data['year'],
                  mpg_data['origin']]).T

mpg_y = mpg_data["mpg"]

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


plt.errorbar(mpg_y_test, mpg_y_hat, yerr=np.sqrt(mpg_V_IJ_unbiased), fmt='o')
plt.plot([5, 45], [5, 45], '--')
plt.show()
