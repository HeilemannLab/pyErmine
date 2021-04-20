#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:00:53 2021

@author: malkusch
"""
import numpy as np


class JumpDistanceModel:
    """
    Univariate Jump-Distance (judi) model

    ...

    Attributes
    ----------
    _diffusion_coefficient : float
        DESCRIPTION. The default is 1.0
    _degrees_of_freedom : int
        DESCRIPTION. The default is 4
    _tau: float
        DESCRIPTION. The default is 0.02
    _mu: float
        Expected value (mean squared displacement).

    Methods
    -------
    pdf(distance)
        Probability density function.
        Calculates the probability of a given jump distance.
    cdf(distance)
        Cumulative density function.
        Calculates the cumulative probability of a given jump distance.
    likelihood(distance)
        Calculates the likelihood of a given distance
    sample(n=1)
        Performs a Monte Carlo sampling of the model.
    """

    def __init__(self, diffusion_coefficient = 1.0, degrees_of_freedom = 4, tau=0.02):
        """
        

        Parameters
        ----------
        diffusion_coefficient : TYPE, optional
            DESCRIPTION. The default is 1.0.
        degrees_of_freedom : TYPE, optional
            DESCRIPTION. The default is 4.
        tau : TYPE, optional
            DESCRIPTION. The default is 0.02.

        Returns
        -------
        None.

        """
        self._diffusion_coefficient = diffusion_coefficient
        self._degrees_of_freedom = degrees_of_freedom
        self._tau = tau
        self._mu = self._degrees_of_freedom * self._diffusion_coefficient * self._tau 
    
    def pdf(self, distance):
        """
        

        Parameters
        ----------
        distance : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        p = ((2 * distance)/self._mu) * np.exp((-np.square(distance))/self._mu)
        return(p)

    def cdf(self, distance):
        """
        

        Parameters
        ----------
        distance : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        p = 1.0-np.exp((-np.square(distance))/self._mu)
        return(p)
    
    def cdf_inverse(self, u):
        """
        

        Parameters
        ----------
        u : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        x = np.sqrt(-1.0 * self._mu * np.log(1.0-u))
        return(x)
    
    def likelihood(self, distance):
        """
        

        Parameters
        ----------
        distance : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        return(self.pdf(distance))
    
    def sample(self, n=1):
        """
        

        Parameters
        ----------
        n : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        """
        u = np.random.uniform(low=0.0, high=1.0, size=n)
        x = self.cdf_inverse(u)
        return(x)