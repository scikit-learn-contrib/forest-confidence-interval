from __future__ import print_function
import sys, os
from setuptools import setup, find_packages
from setuptools.extension import Extension

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

# Get version and release info, which is all stored in forestci/version.py
ver_file = os.path.join('forestci', 'version.py')
with open(ver_file) as f:
    exec(f.read())

ext = Extension(
    'forestci.cyfci',
    ['forestci/cyfci.pyx'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3', '-fopenmp'],
    extra_link_args=['-fopenmp'])

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=find_packages(),
            install_requires=INSTALL_REQUIRES,
            ext_modules=[ext])

if __name__ == '__main__':
    setup(**opts)
