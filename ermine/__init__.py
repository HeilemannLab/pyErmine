#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:12:27 2021

@project: pyErmine
@author: Sebastian Malkusch
@email: malkusch@med.uni-frankfurt.de
@brief: The python package pyErmine analyzes the mobility of laterally diffusing molecules,
such as membrane receptors, using hidden Markov models.
It maps the movements of individual receptors to discrete diffusion states,
all of which are Brownian in nature. The model is trained with single-particle tracking data.

https://github.com/SMLMS/pyErmine
"""

from .preprocessing import *
from .models import *
from .postprocessing import *

name = "ermine"

