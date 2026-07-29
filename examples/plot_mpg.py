"""
======================================
Plotting Regression Forest Error Bars
======================================

This example demonstrates using `forestci` to calculate the error bars of
the predictions of a :class:`sklearn.ensemble.RandomForestRegressor` object.

The data used here are the Auto MPG dataset by R. Quinlan, bundled under the
Creative Commons Attribution 4.0 International license. See ``data/README.md``
for attribution and licensing details.
"""

# Regression Forest Example
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection as xval
import forestci as fci

# Load the bundled Auto MPG data
data_path = Path.cwd() / "data" / "auto_mpg.csv"
if not data_path.exists():
    # Also support running ``python examples/plot_mpg.py`` from the repo root.
    data_path = Path.cwd() / "examples" / "data" / "auto_mpg.csv"
mpg_data = np.genfromtxt(
    data_path,
    delimiter=",",
    skip_header=1,
)

# Separate the predictors and target, removing rows with missing values
mpg_data = mpg_data[~np.isnan(mpg_data).any(axis=1)]
mpg_X = mpg_data[:, :-1]
mpg_y = mpg_data[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = xval.train_test_split(
    mpg_X,
    mpg_y,
    test_size=0.25,
    random_state=42,
)

# Create RandomForestRegressor
n_trees = 2000
forest = RandomForestRegressor(n_estimators=n_trees, random_state=42)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
target_range = [mpg_y.min(), mpg_y.max()]

# Plot predictions without error bars
plt.scatter(y_test, y_pred)
plt.plot(target_range, target_range, 'k--')
plt.xlabel('Reported MPG')
plt.ylabel('Predicted MPG')
plt.show()

# Calculate the variance
variance = fci.random_forest_error(forest, X_train.shape, X_test)

# Plot error bars for predictions using unbiased variance
plt.errorbar(y_test, y_pred, yerr=np.sqrt(variance), fmt='o')
plt.plot(target_range, target_range, 'k--')
plt.xlabel('Reported MPG')
plt.ylabel('Predicted MPG')
plt.show()
