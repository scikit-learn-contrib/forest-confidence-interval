"""
===========================
Plotting something
===========================

Explanation explanation explanation [Reference2001]_

.. [Reference2001] Author, A., Author, B. (2001). Title of the paper.
   Journal of important results 1: 1



"""
import numpy as np
from sklforestci import calc_inbag, random_forest_error
from matplotlib import pyplot as plt

X = np.arange(100).reshape(100, 1)
y = np.random.randn(100)
plt.plot(X, y)
plt.show()


plt.figure()
plt.plot(X, np.random.rand(100))
plt.show()
