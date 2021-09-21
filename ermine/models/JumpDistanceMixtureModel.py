#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:03:02 2021

@project: pyErmine
@author: Sebastian Malkusch
@email: malkusch@med.uni-frankfurt.de
"""
import numpy as np
from .JumpDistanceModel import JumpDistanceModel
from numpy.typing import ArrayLike

class JumpDistanceMixtureModel:
    """ Model mixture of multiple univariate Jump-Distance (judi) distributions and their EM estimation
    
    Attributes
    ----------
    _n_components: int
        Number of mobility modes.
    _degrees_of_freedom: int
        Degrees of freedom for translational movement.
    _tau: float
        Time interval between two consecutive measurements.
    _mu: ArrayLike
        Expected mean squared displacement.
    _weights: ArrayLike
        Population weights of the mobility modes.
    _logLikelihood: float
        Log likelihood of the model.
    _init_params: str
        Initialization parameter argument:
        "m" initialize with radnom values of mu.
        "w" initialize with equal weighted mobility modes.
    _params: str
        Optimization parameter argument:
        "m" optimize values of mu.
        "w" optimize weights mobility modes.
    
    Methods
    ----------
    diffusion_coefficients()
        Calculates the diffusion coefficients that are associated to the respective mean squared displacements.
    fit(x, n_iter, tolerance)
        Parameterization of the model based on the sample of observed jump distances.
    predict(x)
        Predicts the mobility states of the jump distances using Bayesian decision theory.
    fit_predict(x, n_iter, tolerance)
        Parameterization of the model based on the sample of observed jump distances.
        Then, the mobility states of the jump distances are predicted using Bayesian decision theory.
    evaluate(x)
        Evaluate the the model quaity for a given sample of observed jump distances.
    sample(n)
        Generates a random sample of n jump distance observations using the method of Monte-Carlo.
    
    """
    def __init__ (self, n_components: int = 2, degrees_of_freedom: int = 4, tau: float = 0.02, init_params: str = "wm", params:str = "wm"):
        """
        Constructor of the JumpDistanceMixtureModel class.

        Parameters
        ----------
        n_components : int, optional
            Number of mobility modes. The default is 2.
        degrees_of_freedom : int, optional
            Degrees of freedom for translational movement. The default is 4.
        tau : float, optional
            Time interval between two consecutive measurements. The default is 0.02.
        init_params : str, optional
            Parameter argument for initialization step. The default is "wm".
        params : str, optional
            Parameter argument for optimization steps. The default is "wm".

        Returns
        -------
        None.

        """
        self._n_components = n_components
        self._degrees_of_freedom = degrees_of_freedom
        self._tau = tau
        self._mu = np.zeros(([self._n_components]))
        self._weights = np.zeros(([self._n_components]))
        self._logLikelihood = 0
        self._init_params = str(init_params)
        self._params = str(params)
        
    @property
    def n_components(self) -> int:
        """
        Returns the instance variable _n_components

        Returns
        -------
        int
            Number of mobility modes in the model.

        """
        return(int(self._n_components))
    
    @n_components.setter
    def n_components(self, value: int):
        """
        Sets the instance variable _n_components

        Parameters
        ----------
        value : int
            Number of mobility modes in the model.

        Returns
        -------
        None.

        """
        self._n_components = float(value)
        
        
    @property
    def degrees_of_freedom (self) -> int:
        """
        Returns the instance variable _degrees_of_freedom.

        Returns
        -------
        int
            Translational degrees of freedom.

        """
        return (int(self._degrees_of_freedom))

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
    def mu(self) -> ArrayLike:
        """
        Returns the instance variable _mu.

        Returns
        -------
        ArrayLike
            Expected mean squared displacements of the model.

        """
        return(self._mu)
    
    @mu.setter
    def mu(self, value: ArrayLike):
        """
        Sets the instance variable _mu.

        Parameters
        ----------
        value : ArrayLike
            Expected mean squared displacements of the model.

        Returns
        -------
        None.

        """
        self._mu = value
        
    @property
    def weights(self) -> ArrayLike:
        """
        Returns the instance variable _weights.

        Returns
        -------
        ArrayLike
            Weights of the model modes.

        """
        return(self._weights)
    
    @weights.setter
    def weights(self, value: ArrayLike):
        """
        Sets the instance variable _weights.

        Parameters
        ----------
        value : ArrayLike
            Weights of the model modes.

        Returns
        -------
        None.

        """
        self._weights = value
        
    @property
    def init_params(self) -> str:
        """
        Returns the instance variable _init_params.

        Returns
        -------
        str
            Initialization parameter argument:
            "m" initialize with radnom values of mu.
            "w" initialize with equal weighted mobility modes.

        """
        return(self._init_params)
    
    @init_params.setter
    def init_params(self, value: str):
        """
        Sets the instance variable _init_params.

        Parameters
        ----------
        value : str
            Initialization parameter argument:
            "m" initialize with radnom values of mu.
            "w" initialize with equal weighted mobility modes.

        Returns
        -------
        None.

        """
        self._init_params = str(value)
        
    @property
    def params(self) -> str:
        """
        Returns the instance variable _params.

        Returns
        -------
        str
           Optimization parameter argument:
           "m" optimize values of mu.
           "w" optimize weights mobility modes.

        """
        return(self._params)
    
    @params.setter
    def params(self, value: str):
        """
        Sets the instance variable _params.

        Parameters
        ----------
        value : str
            Optimization parameter argument:
            "m" optimize values of mu.
            "w" optimize weights mobility modes.

        Returns
        -------
        None.

        """
        self._params = str(value)
        
    @property
    def logLikelihood(self) -> float:
        """
        Returns the instance variable _logLikelihood.

        Returns
        -------
        float
            The log likelihood of the model.

        """
        return(self._logLikelihood)

        
        
    def diffusion_coefficients(self) -> ArrayLike:
        """
        Calculates the diffusion coefficients that are associated to the respective mean squared displacements.

        Returns
        -------
        ArrayLike
            Array of diffusion coefficients.

        """
        diff_coeff = self._mu/(self._degrees_of_freedom * self._tau)
        return(np.expand_dims(diff_coeff, axis=1))
        
    def pdf(self, x: ArrayLike, mu: float) -> ArrayLike:
        """
        Probability of observing a given jump distance by a model
        characterized by the expected mean squared displacement mu.

        Parameters
        ----------
        x : ArrayLike
            Sample of observed jump distances.
        mu : float
            Expected mean squared displacement.

        Returns
        -------
        ArrayLike
            Observation probability.

        """
        p = ((2 * x)/mu) * np.exp((-np.square(x))/mu)
        return p

    def pdf_super_pos(self, x: ArrayLike) -> ArrayLike:
        """
        Probability of observiang a given jump distance.

        Parameters
        ----------
        x : ArrayLike
            Sample of observed jump distances.

        Returns
        -------
        ArrayLike
            Observation probability.

        """
        probability = np.zeros([np.shape(x)])
        for i in range(self._n_components):
            probability[:] += self._weights[i] * self.pdf(x, mu = self._mu[i])
        return(probability)
        
    def _init_step(self, x: ArrayLike):
        """
        Initialization step used by the the fit method.

        Parameters
        ----------
        x : ArrayLike
            Sample of observed jump distances.

        Returns
        -------
        None.

        """
        mu_min = min(x)
        mu_max = max(x)
        if 'm' in self._init_params:
            u = np.sort(np.random.uniform(low=mu_min, high=mu_max, size=self._n_components))
            self._mu = np.square(u)
        if 'w' in self._init_params:
            self._weights = np.repeat(1/self._n_components, self._n_components)
    
    def _e_step(self, x: ArrayLike):
        """
        Perform the E-step of EM algorithm.

        Parameters
        ----------
        x : ArrayLike
            Sample of observed jump distances.

        Returns
        -------
        None.

        """
        probability = np.zeros([np.shape(x)[0], self._n_components])
        for i in range(self._n_components):
            probability[:,i] = self.pdf(x, self._mu[i]) * self._weights[i]
        denominator = np.sum(probability, axis=1)
        self._normalized_probability_ = np.zeros([np.shape(x)[0], self._n_components])
        for i in range(self._n_components):
            self._normalized_probability_[:,i] = probability[:,i] / denominator[:]
        self._logLikelihood = np.sum(np.log(denominator))

    def _m_step(self, x: ArrayLike):
        """
        Perform the M-step of EM algorithm.

        Parameters
        ----------
        x : ArrayLike
            Sample of observed jump distances.

        Returns
        -------
        None.

        """
        for i in range(self._n_components):
            denominator = np.sum(self._normalized_probability_[:,i])
            if 'm' in self._params:
                self._mu[i] = np.sum(self._normalized_probability_[:,i] * np.square(x[:]))/denominator
            if 'w' in self._params:
                self._weights[i] = denominator / np.shape(x)[0]
                self._weights = self._weights[:] /np.sum(self._weights)
        
    def fit(self, x: ArrayLike, n_iter: int = 1000, tolerance: float =  1e-9):
        """
        Parameterization of the model based on the sample of observed jump distances.
        An initialization step is performed before entering the EM algorithm.
        If you want to avoid this step for a subset of the parameters,
        pass proper init_params keyword argument to JumpDistanceMixtureModel's constructor.
        By default, the mu and weights parameters are optimized.
        If you want to keep a specific model parameter fix,
        pass proper params keyword argument to JumpDistanceMixtureModel's constructor.

        Parameters
        ----------
        x : ArrayLike
            Sample of observed jump distances.
        n_iter : int, optional
            Maximum number of iterations. The default is 1000.
        tolerance : float, optional
            Termination criterion. The default is 1e-9.

        Returns
        -------
        None.

        """
        self._init_step(x)
        self._e_step(x)
        for i in range(n_iter):
            ll_preceding = self._logLikelihood
            self._m_step(x)
            self._e_step(x)
            if np.abs(self._logLikelihood - ll_preceding) < tolerance: break
        
    def predict(self, x: ArrayLike) -> ArrayLike:
        """
        Predicts the mobility states of the jump distances using Bayesian decision theory.

        Parameters
        ----------
        x : ArrayLike
            Sample of observed jump distances.

        Returns
        -------
        ArrayLike
            Predicted mobility states for the observed jump distances.

        """
        p_bayes = np.zeros([np.shape(x)[0], self._n_components])
        denominator = 0
        for j in range(self._n_components):
                denominator += self.pdf(x, self._mu[j]) * self._weights[j]
        for i in range(self._n_components):
            p_bayes[:,i] = self.pdf(x, self._mu[i]) * self._weights[i]/denominator
        y = np.argmax(p_bayes, axis=1)
        return(y)
    
    def fit_predict(self, x: ArrayLike, n_iter: int = 1000, tolerance: float = 1e-9) -> ArrayLike:
        """
        Parameterization of the model based on the sample of observed jump distances.
        Then, the mobility states of the jump distances are predicted using Bayesian decision theory.
        An initialization step is performed before entering the EM algorithm.
        If you want to avoid this step for a subset of the parameters,
        pass proper init_params keyword argument to JumpDistanceMixtureModel's constructor.
        If you want to keep a specific model parameter fix,
        pass proper params keyword argument to JumpDistanceMixtureModel's constructor.

        Parameters
        ----------
        x : ArrayLike
            Sample of observed jump distances.
        n_iter : int, optional
            Maximum number of iterations. The default is 1000.
        tolerance : float, optional
            Termination criterion. The default is 1e-9.

        Returns
        -------
        ArrayLike
            Predicted mobility states for the observed jump distances.

        """
        self.fit(x, n_iter = 1000, tolerance =  1e-9)
        y = self.predict(x)
        return(y)
    
    def evaluate(self, x: ArrayLike) -> dict:
        """
        Evaluate the the model quaity for a given sample of observed jump distances.

        Parameters
        ----------
        x : ArrayLike
            Sample of observed jump distances.

        Returns
        -------
        dict
            Evaluation metrics of observation.

        """
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
    
    def sample(self, n: int=1) -> (ArrayLike, ArrayLike):
        """
        Generates a random sample of n jump distance observations using the method of Monte-Carlo.

        Parameters
        ----------
        n : int, optional
            Sample size. The default is 1.

        Returns
        -------
        (ArrayLike, ArrayLike)
            Generated jump distance observations.
            State sequence.

        """
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