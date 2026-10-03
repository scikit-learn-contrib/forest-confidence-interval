.. _prediction_error:

Confidence intervals versus observed prediction error
=====================================================

Each point below is a held-out observation. The horizontal coordinate is the
absolute residual, :math:`|y_i - \hat y_i|`; the vertical coordinate is the
approximate 95% confidence-interval half-width,
:math:`1.96\sqrt{\max(\widehat{V}_{IJ,i}, 0)}`. All models use 2,000 estimators
and ``calibrate=False``. Negative raw IJ variances are clipped only to compute
the square root; each caption reports their number. A zero width caused by
clipping is not evidence of certainty.

The dashed diagonal marks equal magnitudes. Points above it have observed
errors smaller than the estimated half-width; points below it have larger
errors. This is a diagnostic comparison, not a prediction-interval coverage
validation: IJ uncertainty describes variation of the fitted prediction under
training-set resampling. Observed residuals also contain irreducible outcome
noise and model bias, so a nominal 95% confidence interval need not contain
95% of individual outcomes. The observed residual is not the unknown error
relative to the true conditional mean.

For regression, the prediction is the mean of the individual estimator
predictions. For binary classification, ``forestci`` uses individual trees'
0/1 class predictions, so we compare their mean vote fraction with the observed
0/1 outcome. This matches the quantity used internally by the IJ computation;
it is not ``RandomForestClassifier.predict_proba`` or a hard-class error rate.
The normal approximation is descriptive and is not clipped to [0, 1].

The first six panels use the same datasets, 80/20 splits, and reference forests
as :doc:`calibration_benchmark`. Auto MPG covers the random-forest regression
gallery example, using the benchmark's 20% test split rather than the gallery's
25%. Additional panels cover the spam classifier (20% test split) and Auto MPG
bagged SVR (25% test split), with their gallery model settings except that the
ensemble size is increased to 2,000. All data-generation, split and model seeds
are 42. California Housing uses all 20,640 rows.

Every documentation build executes ``examples/generate_calibration_benchmark.py``
to refit the models and regenerate these figures and the calibration table.
No saved benchmark results or plot images are used as build inputs.

.. include:: generated/benchmarks/prediction_error_figures.rst
