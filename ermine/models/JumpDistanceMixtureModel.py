#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:03:02 2021

@author: malkusch
"""
import numpy as np
from .JumpDistanceModel import JumpDistanceModel

class JumpDistanceMixtureModel:
    "Model mixture of multiple univariate Jump-Distance (judi) distributions and their EM estimation"
    def __init__ (self, n_components = 2, degrees_of_freedom = 4, tau=0.02):
        self._n_components = n_components
        self._degrees_of_freedom = degrees_of_freedom
        self._tau = tau
        self._mu = np.zeros(([self._n_components]))
        self._weights = np.zeros(([self._n_components]))
        self._logLikelihood = 0
        
    def diffusion_coefficients(self):
        diff_coeff = self._mu/(self._degrees_of_freedom * self._tau)
        return(np.expand_dims(diff_coeff, axis=1))
        
    def pdf(self, x, mu):
        "Probability of a data point given the current parameters"
        p = ((2 * x)/mu) * np.exp((-np.square(x))/mu)
        return p

    def pdf_super_pos(self, x):
        probability = np.zeros([np.shape(x)])
        for i in range(self._n_components):
            probability[:] += self._weights[i] * self.pdf(x, mu = self._mu[i])
        return(probability)
        
    def _init_step(self, x):
        mu_min = min(x)
        mu_max = max(x)
        u = np.sort(np.random.uniform(low=mu_min, high=mu_max, size=self._n_components))
        self._mu = np.square(u)
        self._weights = np.repeat(1/self._n_components, self._n_components)
    
    def _e_step(self, x):
        probability = np.zeros([np.shape(x)[0], self._n_components])
        for i in range(self._n_components):
            probability[:,i] = self.pdf(x, self._mu[i]) * self._weights[i]
        denominator = np.sum(probability, axis=1)
        self._normalized_probability_ = np.zeros([np.shape(x)[0], self._n_components])
        for i in range(self._n_components):
            self._normalized_probability_[:,i] = probability[:,i] / denominator[:]
        self._logLikelihood = np.sum(np.log(denominator))

    def _m_step(self, x):
        for i in range(self._n_components):
            denominator = np.sum(self._normalized_probability_[:,i])
            self._mu[i] = np.sum(self._normalized_probability_[:,i] * np.square(x[:]))/denominator
            self._weights[i] = denominator / np.shape(x)[0]
        self._weights = self._weights[:] /np.sum(self._weights)
        
    def fit(self, x, n_iter = 1000, tolerance =  1e-9):
        self._init_step(x)
        self._e_step(x)
        for i in range(n_iter):
            ll_preceding = self._logLikelihood
            self._m_step(x)
            self._e_step(x)
            if np.abs(self._logLikelihood - ll_preceding) < tolerance: break
        
    def predict(self, x):
        p_bayes = np.zeros([np.shape(x)[0], self._n_components])
        denominator = 0
        for j in range(self._n_components):
                denominator += self.pdf(x, self._mu[j]) * self._weights[j]
        for i in range(self._n_components):
            p_bayes[:,i] = self.pdf(x, self._mu[i]) * self._weights[i]/denominator
        y = np.argmax(p_bayes, axis=1)
        return(y)
    
    def fit_predict(self, x, n_iter = 1000, tolerance =  1e-9):
        self.fit(x, n_iter = 1000, tolerance =  1e-9)
        y = self.predict(x)
        return(y)
    
    def evaluate(self, x):
        self._e_step(x)
        dof = self._n_components + self._n_components - 1
        instances = np.shape(x)[0]
        bic = dof * np.log(instances) - 2 * self._logLikelihood
        aic = 2 * dof - 2 * self._logLikelihood
        aicc = aic + (2*np.square(dof) + 2 * dof)/(instances - dof - 1)
        dictionary = {"classes": [self._n_components],
                      "dof": [dof],
                      "instances": [instances],
                      "log_likelihood": [self._logLikelihood],
                      "BIC": [bic],
                      "AIC": [aic],
                      "AICc": [aicc]}
        return(dictionary)
    
    def sample(self, n):
        judi = JumpDistanceModel(diffusion_coefficient = self.diffusion_coefficients()[0],
                                 degrees_of_freedom = self._degrees_of_freedom,
                                 tau = self._tau)
        j = int(np.ceil(n * self._weights[0]))
        x = judi.simulate(j)
        y = np.repeat(0, j)
        if self._n_components > 1:
            for i in np.arange(start = 1, stop = self._n_components, step = 1):
                judi = JumpDistanceModel(diffusion_coefficient = self.diffusion_coefficients()[i],
                                 degrees_of_freedom = self._degrees_of_freedom,
                                 tau = self._tau)
                j = int(np.ceil(n * self._weights[i]))
                x = np.append(x, judi.simulate(j))
                y = np.append(y, np.repeat(i, j))
                
        return(x,y)