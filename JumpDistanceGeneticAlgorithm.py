#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 19:55:20 2021

@author: malkusch
"""
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from .JumpDistanceModel import JumpDistanceModel

class JumpDistanceGeneticAlgorithm:
    
    def __init__(self, n_components = 2, degrees_of_freedom = 4, tau=0.02):
        self._n_components = n_components
        self._degrees_of_freedom = degrees_of_freedom
        self._tau = tau
        self._mu = np.zeros(([self._n_components]))
        self._weights = np.zeros(([self._n_components]))
        self._logLikelihood = 0
        self._varbound=np.array([[[0.0,1.0]] * self._n_components, [[0,1]] * self._n_components])
        self._varbound = np.reshape(self._varbound, newshape=(self._n_components * 2, 2))

       
        
    def diffusion_coefficients(self):
        return(self._mu/(self._degrees_of_freedom * self._tau))
        
    def pdf(self, x, mu):
        "Probability of a data point given the current parameters"
        p = ((2 * x)/mu) * np.exp((-np.square(x))/mu)
        return p

    def pdf_super_pos(self, x):
        probability = np.zeros([np.shape(x)][0])
        for i in range(self._n_components):
            probability[:] += self._weights[i] * self.pdf(x, mu = self._mu[i])
        return(probability)
        
     
    def _objective(self, parameter):
        omega = parameter[:self._n_components]
        omega = omega / np.sum(omega)
        mu = parameter[self._n_components:]
        probability = np.zeros([np.shape(self._x)][0])
        for i in range(self._n_components):
            probability[:] += omega[i] * self.pdf(x = self._x, mu = mu[i])
        return(np.sum(np.log(probability)))
    
    
    def fit(self, x):
        self._x = x
        self._varbound[self._n_components:, :] = [0, np.max(self._x)]
        self._hyper_parameter = {'max_num_iteration': 50,
                                  'population_size': 100,
                                  'mutation_probability': 0.1,
                                  'elit_ratio': 0.01,
                                  'crossover_probability': 0.5,
                                  'parents_portion': 0.3,
                                  'crossover_type': 'uniform',
                                  'max_iteration_without_improv': None}
        model = ga(function=self._objective,
                   dimension= 2*self._n_components,
                   variable_type='real',
                   variable_boundaries=self._varbound,
                   algorithm_parameters = self._hyper_parameter)
        model.run()
        best_model = model.output_dict["variable"]
        self._weights = best_model[:self._n_components] / np.sum(best_model[:self._n_components])
        self._mu = best_model[self._n_components:]
        self._logLikelihood = -np.sum(np.log(self.pdf_super_pos(x)))
        self._x = []
        
    def evaluate(self, x):
        self._logLikelihood = -np.sum(np.log(self.pdf_super_pos(x)))
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
    
    def predict(self, x):
        p_bayes = np.zeros([np.shape(x)[0], self._n_components])
        denominator = 0
        for j in range(self._n_components):
                denominator += self.pdf(x, self._mu[j]) * self._weights[j]
        for i in range(self._n_components):
            p_bayes[:,i] = self.pdf(x, self._mu[i]) * self._weights[i]/denominator
        y = np.argmax(p_bayes, axis=1)
        return(y)
    
    def fit_predict(self, x):
        self.fit(x)
        y = self.predict(x)
        return(y)
    
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