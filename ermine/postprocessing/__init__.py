#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 09:36:20 2021

@project: pyErmine
@author: Sebastian Malkusch
@email: malkusch@med.uni-frankfurt.de
@brief: The module contains post-processing functions for the interpretation of modeling results.
Module functions are:
    calculate_diffusion_coefficient
    calculate_expectation_value
"""
from .error_correction import calculate_diffusion_coefficient
from .error_correction import calculate_expectation_value
from .error_correction import static_error

__all__=["calculate_diffusion_coefficient",
         "calculate_expectation_value",
         "static_error"]