#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:07:25 2021

@project: pyErmine
@author: Sebastian Malkusch
@email: malkusch@med.uni-frankfurt.de
"""

import numpy as np


def calculate_expectation_value(diff_coeff: float, tau: float = 0.02, sigma: float = 0.02, epsilon: float = 10.0) -> float:
    """
    Calculates the expected mean squared displacement based upon the
    diffusion coefficient and the static and dynamic measurement errors.
    The calculation is done according to
    Savin T. and Doyle S.,
    Static and Dynamic Errors in Particle Tracking Microrheology
    Biophysical Journal 2005

    Parameters
    ----------
    diff_coeff : float
        corrected diffusion coefficnet.
    tau : float, optional
        Time difference between measurements. The default is 0.02.
    sigma : float, optional
        Integration time of a measurement. The default is 0.02.
    epsilon : float, optional
        Localization precsison. The default is 10.0.

    Returns
    -------
    msd : float
        Expected mean squared displacement.

    """
    msd = 4*diff_coeff * (tau - (sigma/3.0)) + 4 * np.square(epsilon)
    return msd


def calculate_diffusion_coefficient(expected_value: float, tau: float = 0.02, sigma: float = 0.02, epsilon: float = 10.0) -> float:
    """
    Calculates the expected diffusion coefficient
    corrected for static and dynamic errors.
    The calculation is done according to
    Savin T. and Doyle S.,
    Static and Dynamic Errors in Particle Tracking Microrheology
    Biophysical Journal 2005

    Parameters
    ----------
    expected_value : float
        Measured mean squared displacement
    tau : float, optional
        Time difference between measurements. The default is 0.02.
    sigma : float, optional
        Integration time of a measurement. The default is 0.02.
    epsilon : float, optional
        Localization precsison. The default is 10.0.

    Returns
    -------
    diff_coeff : float
        Corrected diffusion coefficnet.

    """
    diff_coeff = (expected_value - 4.0 * np.square(epsilon)) / (4.0 * tau - (4.0/3.0) * sigma) 
    return diff_coeff

def static_error(apparent_msd_d0: float) -> float:
    """
    Calculates the static error of localization of a fixed molecule
    from the apparent mean square displacement.
    The calculation is done according to
    Savin T. and Doyle S.,
    Static and Dynamic Errors in Particle Tracking Microrheology
    Biophysical Journal 2005

    Parameters
    ----------
    apparent_msd_d0 : float
        Measured mean squared displacement of a fix molecule.

    Returns
    -------
    epsilon: float
        Expected localization precsison..

    """
    return(np.sqrt(apparent_msd_d0/4.0))
