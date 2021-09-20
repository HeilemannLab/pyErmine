#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:20:11 2021

@author: malkusch
"""
import pathlib
from setuptools import setup, find_packages


HERE = pathlib.Path(__file__).parent
# The text of the README file
README = (HERE / "README.md").read_text()

setup(name='pyErmine',
      python_requires='>=3.5',
      version='0.1.2',
      description='Estimate Reaction-rates by Markov-based Investigation of Nanoscopy Experiments (ermine) using Python.',
      long_description=README,
      long_description_content_type="text/markdown",
      url='https://github.com/SMLMS/pyErmine',
      author='Sebastian Malkusch',
      author_email='malkusch@med.uni-frankfurt.de',
      license='GNU General Public License v3 (GPLv3)',
      packages = find_packages(),
      zip_safe=False,
      install_requires=['hmmlearn>=0.2.4',
                        'numpy>=1.19.2',
                        'pandas>=1.1.5',
                        'scikit-learn>=0.23.2'],
      keywords=['hidden markov model', 'unsupervised learning', 'single particle tracking', 'biophysics'],
      classifiers= [
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux"
        ])
