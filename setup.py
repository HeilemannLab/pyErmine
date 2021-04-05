#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:20:11 2021

@author: malkusch
"""
from setuptools import setup, find_packages

setup(name='ermine',
      version='0.1',
      description='Estimate Reaction-rates by Markov-based Investigation of Nanoscopy Experiments (ermine) using Python.',
      url='https://github.com/SMLMS/pyErmine',
      author='Sebastian Malkusch',
      author_email='malkusch@med.uni-frankfurt.de',
      license='GNU General Public License v3 (GPLv3)',
      packages = find_packages(),
      zip_safe=False,
      install_requires=['hmmlearn', 'matplotlib', 'numpy', 'pandas', 'scipy', 'sklearn'],
      keywords=['hidden markov models', 'unsupervised learning', 'single particle diffucion', "biophysics"],
      classifiers= [
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux"
        ])