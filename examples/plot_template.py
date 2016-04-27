"""
===========================
Plotting something
===========================

An example plot

"""
import numpy as np
from sklforestci import calc_inbag, random_forest_error
from matplotlib import pyplot as plt

X = np.arange(100).reshape(100, 1)
y = np.random.randn((100, ))
plt.plot(x, y)
plt.show()
