#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:00:53 2021

@author: malkusch
"""
import numpy as np


class JumpDistanceModel:
    "Univariate Jump-Distance (judi) model"
    def __init__(self, diffusion_coefficient = 1.0, degrees_of_freedom = 4, tau=0.02):
        self._diffusion_coefficient = diffusion_coefficient
        self._degrees_of_freedom = degrees_of_freedom
        self._tau = tau
        self._mu = self._degrees_of_freedom * self._diffusion_coefficient * self._tau 
    
    def pdf(self, distance):
        p = ((2 * distance)/self._mu) * np.exp((-np.square(distance))/self._mu)
        return(p)

    def cdf(self, distance):
        p = 1.0-np.exp((-np.square(distance))/self._mu)
        return(p)
    
    def cdf_inverse(self, u):
        x = np.sqrt(-1.0 * self._mu * np.log(1.0-u))
        return(x)
    
    def likelihood(self, distance):
        return(self.pdf(distance))
    
    def sample(self, n=1):
        u = np.random.uniform(low=0.0, high=1.0, size=n)
        x = self.cdf_inverse(u)
        return(x)