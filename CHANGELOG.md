# Changelog

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
