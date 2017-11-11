# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 3
_version_micro = ''  # use '' for first of series, number for 1 and above
# _version_extra = 'dev'
_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "forestci: confidence intervals for scikit-learn "
description += "forest algorithms"
# Long description will go up on the pypi page
long_description = """

sklearn forest ci
=================

`forest-confidence-interval` is a Python module for calculating variance and
adding confidence intervals to scikit-learn random forest regression or
classification objects. The core functions calculate an in-bag and error bars
for random forest objects

Please read the repository README_ on Github or our documentation_

.. _README: https://github.com/scikit-learn-contrib/forest-confidence-interval/blob/master/README.md

.. _documentation: http://contrib.scikit-learn.org/forest-confidence-interval/

"""

NAME = "forestci"
MAINTAINER = "Ariel Rokem"
MAINTAINER_EMAIL = "arokem@uw.edu"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/scikit-learn-contrib/forest-confidence-interval"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Ariel Rokem, Bryna Hazelton, Kivan Polimis"
AUTHOR_EMAIL = "arokem@uw.edu"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
