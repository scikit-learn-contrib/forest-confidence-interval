# Changelog

## Unreleased

- Add opt-in class-probability IJ variance via `class_index`, preserving binary
  vote-fraction behavior and warning on legacy multiclass hard-label variance.
  Add Wine multiclass examples and per-class calibration/error diagnostics.

- Remove duecredit integration and preserve scientific citations in docstrings.
- Remove unused coverage configuration and the obsolete root GitHub Pages marker.

- Add optional progress reporting for memory-constrained variance
  calculations.

- Improvement of calibration method, with proper documentation and
  six benchmarks.

- fix `inbag[0:n_trees]` improperly trimming the matrix in the case 
  n_train > n_trees, for the calculation of the bias correction

## 0.8 (2026-07-25)

The main feature of this release is compatibility with scikit-learn 1.9.

- Support the `sample_weight` signature used by scikit-learn 1.9 forest
  internals.
- Test against the latest bugfix release of every scikit-learn minor series
  from 1.0 through 1.9, plus the latest release below 2.0.
- Support Python 3.12 and pytest 9.
- Adopt a modern `pyproject.toml` build and `src/` package layout.
- Publish reproducible documentation through GitHub Pages without relying on
  external example datasets.
