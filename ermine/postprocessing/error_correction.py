#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:07:25 2021

@project: pyErmine
@author: Sebastian Malkusch
@email: malkusch@med.uni-frankfurt.de
"""

import numpy as np
from numpy.typing import ArrayLike


def calculate_expectation_value(diff_coeff: ArrayLike, tau: float = 0.02, dof: int=4,  sigma: float = 0.02, epsilon: float = 10.0) -> ArrayLike:
    """
    Calculates the expected mean squared displacement based upon the
    diffusion coefficient and the static and dynamic measurement errors.
    The calculation is done according to
    Savin T. and Doyle S.,
    Static and Dynamic Errors in Particle Tracking Microrheology
    Biophysical Journal 2005

    Parameters
    ----------
    diff_coeff : Arraylike
        corrected diffusion coefficnet.
    tau : float, optional
        Time difference between measurements. The default is 0.02.
    dof : int, optinal
        Degrees of freedom for translational mobility. The default is 4.
    sigma : float, optional
        Integration time of a measurement. The default is 0.02.
    epsilon : float, optional
        Localization precsison. The default is 10.0.

    Returns
    -------
    msd : ArrayLike
        Expected mean squared displacement.

    """
    msd = dof*diff_coeff * (tau - (sigma/3.0)) + dof * np.square(epsilon)
    return msd


def calculate_diffusion_coefficient(expected_value: ArrayLike, tau: float = 0.02, dof: int = 4, sigma: float = 0.02, epsilon: float = 10.0) -> ArrayLike:
    """
    Calculates the expected diffusion coefficient
    corrected for static and dynamic errors.
    The calculation is done according to
    Savin T. and Doyle S.,
    Static and Dynamic Errors in Particle Tracking Microrheology
    Biophysical Journal 2005

    Parameters
    ----------
    expected_value : ArrayLike
        Measured mean squared displacement
    tau : float, optional
        Time difference between measurements. The default is 0.02.
    dof : int, optinal
        Degrees of freedom for translational mobility. The default is 4.
    sigma : float, optional
        Integration time of a measurement. The default is 0.02.
    epsilon : float, optional
        Localization precsison. The default is 10.0.

    Returns
    -------
    diff_coeff : ArrayLike
        Corrected diffusion coefficnet.

    """
    diff_coeff = (expected_value - float(dof) * np.square(epsilon)) / (float(dof) * tau - (dof/3.0) * sigma) 
    return diff_coeff

def static_error(apparent_msd_d0: ArrayLike, dof: int = 4) -> ArrayLike:
    """
    Calculates the static error of localization of a fixed molecule
    from the apparent mean square displacement.
    The calculation is done according to
    Savin T. and Doyle S.,
    Static and Dynamic Errors in Particle Tracking Microrheology
    Biophysical Journal 2005

    Parameters
    ----------
    apparent_msd_d0 : ArrayLike
        Measured mean squared displacement of a fix molecule.
    dof: int, optional
        Degrees of freedom for translational mobility. The default is 4.

    Returns
    -------
    epsilon: ArrayLike
        Expected localization precsison..

    """
    return(np.sqrt(apparent_msd_d0/float(dof)))
