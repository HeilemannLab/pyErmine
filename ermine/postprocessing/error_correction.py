#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:07:25 2021

@author: malkusch
"""

import numpy as np


def calculate_expectation_value(diff_coeff, tau = 0.02, sigma = 0.02, epsilon = 10.0):
    """ Calculates the expected mean squared displacement based upon the
    diffusion coefficient and the static and dynamic measurement errors.
    The calculation is done according to
    Savin T. and Doyle S.,
    Static and Dynamic Errors in Particle Tracking Microrheology
    Biophysical Journal 2005
    
    diff_coeff := corrected diffusion coefficnet
    tau := time difference between measurements
    sigma := integration time of a measurement
    epsilon := localization precsison
    """
    msd = 4*diff_coeff * (tau - (sigma/3.0)) + 4 * np.square(epsilon)
    return msd


def calculate_diffusion_coefficient(expected_value, tau = 0.02, sigma = 0.02, epsilon = 10.0):
    """ Calculates the expected diffusion coefficient
    corrected for static and dynamic errors.
    The calculation is done according to
    Savin T. and Doyle S.,
    Static and Dynamic Errors in Particle Tracking Microrheology
    Biophysical Journal 2005
    
    expected_value := mean squared displacement 
    tau := time difference between measurements
    sigma := integration time of a measurement
    epsilon := localization precsison
    """
    diff_coeff = (expected_value - 4.0 * np.square(epsilon)) / (4.0 * tau - (4.0/3.0) * sigma) 
    return diff_coeff

def static_error(apparent_msd_d0):
    return(np.sqrt(apparent_msd_d0/4.0))
