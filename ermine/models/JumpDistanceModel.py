#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:00:53 2021

@project: pyErmine
@author: Sebastian Malkusch
@email: malkusch@med.uni-frankfurt.de
"""
import numpy as np
from numpy.typing import ArrayLike

class JumpDistanceModel:
    """
    Univariate Jump-Distance (judi) model



    Attributes
    ----------
    _diffusion_coefficient : float
        Diffusion coefficient of the mobility state. The default is 1.0
    _degrees_of_freedom : int
        The translational degrees of freedom of particle motion. The default is 4
    _tau: float
        The time interval between two consecutive measurements of particle localization. The default is 0.02
    _mu: float
        Expected value of the mean squared displacement for the given diffusion coefficient.

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

    def __init__(self, diffusion_coefficient: float = 1.0, degrees_of_freedom: int = 4, tau: float = 0.02):
        """
        Constructor of the JumpDistanceModel class.

        Parameters
        ----------
        diffusion_coefficient : float, optional
            Diffusion coefficient of the mobility state. The default is 1.0.
        degrees_of_freedom : int, optional
            The translational degrees of freedom of particle motion. The default is 4.
        tau : float, optional
            The time interval between two consecutive measurements of particle localization. The default is 0.02.

        Returns
        -------
        None.

        """
        self._diffusion_coefficient = diffusion_coefficient
        self._degrees_of_freedom = degrees_of_freedom
        self._tau = tau
        self._mu = self._degrees_of_freedom * self._diffusion_coefficient * self._tau
        
    @property
    def diffusion_coefficient(self) -> float:
        """
        Returns the instance variable _diffusion_coefficient.

        Returns
        -------
        float
            The diffusion coefficient of the model.

        """
        return (self._diffusion_coefficient)

    @diffusion_coefficient.setter
    def diffusion_coefficient(self, value: float):
        """
        Sets the instance parameter _diffusion_coefficient.

        Parameters
        ----------
        value : float
            The diffusion coefficient of the model.

        Returns
        -------
        None.

        """
        self._diffusion_coefficient = float(value)
        
    @property
    def degrees_of_freedom (self) -> int:
        """
        Returns the instance variable _degrees_of_freedom.

        Returns
        -------
        int
            Translational degrees of freedom.

        """
        return (self._degrees_of_freedom)

    @degrees_of_freedom.setter
    def degrees_of_freedom_ (self, value: int):
        """
        Sets the instance variable _degrees_of_freedom.

        Parameters
        ----------
        value : int
            Degrees of freedom for translational movement.

        Returns
        -------
        None.

        """
        self._degrees_of_freedom = float(value)
        
    @property
    def tau(self) -> float:
        """
        Returns the instance variable _tau.

        Returns
        -------
        float
            The time interval between two consecutive measurements.

        """
        return (self._tau)

    @tau.setter
    def tau(self, value: float):
        """
        Sets the instance parameter _tau.

        Parameters
        ----------
        value : float
            The time interval between two consecutive measurements.

        Returns
        -------
        None.

        """
        self._tau = float(value)
        
    @property
    def mu(self) -> float:
        """
        Returns the instance variable _mu.

        Returns
        -------
        float
            Expected mean squared displacement.

        """
        return(self._mu)
    
    def pdf(self, distance: ArrayLike) -> ArrayLike:
        """
        Probability density function for observed jump distances.

        Parameters
        ----------
        distance : ArrayLike
            Observed jump distances.

        Returns
        -------
        ArrayLike
            Probability to observe the given jump distances.

        """
        p = ((2 * distance)/self._mu) * np.exp((-np.square(distance))/self._mu)
        return(p)

    def cdf(self, distance: ArrayLike) -> ArrayLike:
        """
        Cumulative density function for observed jump distances.

        Parameters
        ----------
        distance : ArrayLike
            Observed jump distances.

        Returns
        -------
        ArrayLike
            Cumulative probability to observe the given jump distances.

        """
        p = 1.0-np.exp((-np.square(distance))/self._mu)
        return(p)
    
    def cdf_inverse(self, u: ArrayLike) -> ArrayLike:
        """
        Inverse of the cumulative density function for observed jump distances.

        Parameters
        ----------
        u : ArrayLike
            Cumulative probability to observe the given jump distances.

        Returns
        -------
        ArrayLike
            Jump distance associated with the given cumulative probability.

        """
        x = np.sqrt(-1.0 * self._mu * np.log(1.0-u))
        return(x)
    
    def likelihood(self, distance: ArrayLike) -> ArrayLike:
        """
        Calculates the likelihood for an observed jump distance.

        Parameters
        ----------
        distance : ArrayLike
            Observed jump distances.

        Returns
        -------
        ArrayLike
            likelihood for an observed jump distance.

        """
        return(self.pdf(distance))
    
    def sample(self, n: int = 1) -> ArrayLike:
        """
        Generates a random sample of n jump distance observations using the method of Monte-Carlo.

        Parameters
        ----------
        n : int, optional
            Sample size. The default is 1.

        Returns
        -------
        ArrayLike
            Generated jump distance observations.

        """
        u = np.random.uniform(low=0.0, high=1.0, size=n)
        x = self.cdf_inverse(u)
        return(x)