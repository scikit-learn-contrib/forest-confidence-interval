from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

setup(name='skltreeci',
      version='0.1',
      description='Confidence intervals for scikit-learn forest algorithms',
      author='Ariel Rokem, Bryna Hazelton, Kivan Polimis',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      author_email='arokem@uw.edu',
      )
