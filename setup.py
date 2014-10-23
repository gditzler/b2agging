#!/usr/bin/env python
from __future__ import division
from setuptools import setup

__author__ = "Gregory Ditzler"
__copyright__ = "Copyright 2014, Gregory Ditzler"
__maintainer__ = "Gregory Ditzler"
__license__ = "GPL V3.0"
__version__ = "0.1.0"
__status__ = "development"
__email__ = "gregory.ditzler@gmail.com"


setup(name="b2ag",
      version=__version__,
      description="Estimating error bars for ensemble classifiers.",
      author=__maintainer__,
      author_email=__email__,
      maintainer=__maintainer__,
      maintainer_email=__email__,
      packages=["b2bag"],
      license=__license__,
      keywords=["confidence estimation", 
        "multiple classifer systems", 
        "error bars"],
      platforms=['MacOS', 'Linux']
    )

