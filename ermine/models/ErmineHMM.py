#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:07:25 2021

@project: pyErmine
@author: Sebastian Malkusch
@email: malkusch@med.uni-frankfurt.de
"""
from hmmlearn.base import _BaseHMM

import numpy as np
from sklearn.utils import check_random_state
import numpy as np
from numpy.typing import ArrayLike

class ErmineHMM(_BaseHMM):
    """
    Hidden Markov model with jump distance model emissions.
        
    Attributes
    ----------
    _n_components_ : int, optional
        Number of states. The default is 1.
    _startprob_prior_ : float, optional
        Parameters of the Dirichlet prior distribution for startprob_. The default is 1.0.
    _transmat_prior_ : float, optional
        Parameters of the Dirichlet prior distribution for each row of the transition probabilities transmat_. The default is 1.0.
    _diffusion_degrees_of_freedom_ : int, optional
        Translational degrees of freedom. The default is 4.
    _tau_ : float, optional
        Time interval between two consecutive measurements. The default is 0.02.
    _algorithm_ : str, optional
        Decoder algorithm. The default is "viterbi".
    _random_state_ : int, optional
        A random number generator instance. The default is 42.
    _n_iter_ : int, optional
        Maximum number of iterations to perform. The default is 10.
    _tol_ : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood is below this value. The default is 1e-2.
    _verbose_ : bool, optional
        Whether per-iteration convergence reports are printed to sys.stderr. Convergence can also be diagnosed using the monitor_ attribute. The default is False.
    _params_ : str, optional
        The parameters that get updated during training.
        Can contain any combination of ‘s’ for startprob, ‘t’ for transmat, ‘d’ for diffusion coefficients. Defaults to all parameters. The default is "std".
    _init_params_ : str, optional
        The parameters that get initialized prior to training.
        Can contain any combination of ‘s’ for startprob, ‘t’ for transmat, ‘d’ for diffusion coefficients. Defaults to all parameters. The default is "std".
    _diffusion_coefficients_: ArrayLike
        Diffusion coefficients of mobility modes.
    _mu_: ArrayLike
        Expected mean squared displacements of of mobilty modes.
        
    Methods
    ----------
    fit(self, X: ArrayLike, lengths: ArrayLike)   :
        Parameterization of the model based on the given jump distance sequence.
    evaluate(self, X: ArrayLike, lengths: ArrayLike = None):
        Evaluate the the model quality for a given sample of observed jump distances.
    predict(X, lengths=None):
        Find most likely state sequence corresponding to X.
    
    """
    

    def __init__(self, n_components: int=1,
                 startprob_prior:float=1.0, transmat_prior:float=1.0,
                 diffusion_degrees_of_freedom:int = 4,
                 tau:float = 0.02,
                 algorithm:str="viterbi", random_state:int=42,
                 n_iter:int=10, tol:float=1e-2, verbose:bool=False,
                 params:str="std", init_params:str="std"):
        """
        

        Parameters
        ----------
        n_components : int, optional
            Number of states. The default is 1.
        startprob_prior : float, optional
            Parameters of the Dirichlet prior distribution for startprob_. The default is 1.0.
        transmat_prior : float, optional
            Parameters of the Dirichlet prior distribution for each row of the transition probabilities transmat_. The default is 1.0.
        diffusion_degrees_of_freedom : int, optional
            Translational degrees of freedom. The default is 4.
        tau : float, optional
            Time interval between two consecutive measurements. The default is 0.02.
        algorithm : str, optional
            Decoder algorithm. The default is "viterbi".
        random_state : int, optional
            A random number generator instance. The default is 42.
        n_iter : int, optional
            Maximum number of iterations to perform. The default is 10.
        tol : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood is below this value. The default is 1e-2.
        verbose : bool, optional
            Whether per-iteration convergence reports are printed to sys.stderr. Convergence can also be diagnosed using the monitor_ attribute. The default is False.
        params : str, optional
            The parameters that get updated during training.
            Can contain any combination of ‘s’ for startprob, ‘t’ for transmat, ‘d’ for diffusion coefficients. Defaults to all parameters. The default is "std".
        init_params : str, optional
            The parameters that get initialized prior to training.
            Can contain any combination of ‘s’ for startprob, ‘t’ for transmat, ‘d’ for diffusion coefficients. Defaults to all parameters. The default is "std".

        Returns
        -------
        None.

        """
        _BaseHMM.__init__(self, n_components=n_components, startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose, init_params=init_params)

        self._tau_ = tau
        self._diffusion_degrees_of_freedom_ = diffusion_degrees_of_freedom
        self._diffusion_coefficients_ = np.zeros([self.n_components])
        self._mu_ = np.zeros([self.n_components])
        self.n_features = 1


    @property
    def tau_(self) -> float:
        """
        Returns the instance variable _tau_

        Returns
        -------
        float
            The time interval between two consecutive measurements.

        """
        return (self._tau_)

    @tau_.setter
    def tau_(self, value: float):
        """
        Sets the instance parameter _tau_.

        Parameters
        ----------
        value : float
            The time interval between two consecutive measurements.

        Returns
        -------
        None.

        """
        self._tau_ = float(value)
        
    @property
    def diffusion_degrees_of_freedom_ (self) -> int:
        """
        Returns the instance variable _diffusion_degrees_of_freedom_.

        Returns
        -------
        int
            Translational degrees of freedom.

        """
        return (self._diffusion_degrees_of_freedom_ )

    @diffusion_degrees_of_freedom_.setter
    def diffusion_degrees_of_freedom_ (self, value: int):
        """
        Sets the instance variable _diffusion_degrees_of_freedom..

        Parameters
        ----------
        value : int
            Degrees of freedom for translational movement.

        Returns
        -------
        None.

        """
        self._diffusion_degrees_of_freedom_ = float(value)


        
    @property
    def mu_(self) -> ArrayLike:
        """
        Returns the instance variable _mu_

        Returns
        -------
        ArrayLike
            mu.

        """
        return (self._mu_)
    
    @mu_.setter
    def mu_(self, values: ArrayLike):
        """
        Sets the instance variable _mu_.
        Updates the instance variables _diffusion_coefficients_ and _n_components_.

        Parameters
        ----------
        values : ArrayLike
            mu.

        Returns
        -------
        None.

        """
        self._mu_ = np.array(values, copy=True)
        self._update_diffusion_coefficients()
        self.n_components = self._mu_.shape[0]
        
    @property
    def diffusion_coefficients_(self) -> ArrayLike:
        """
        Returns the instance variable _diffusion_coefficients_

        Returns
        -------
        ArrayLike
            diffusion_coefficients.

        """
        return (self._diffusion_coefficients_)
    
    @diffusion_coefficients_.setter
    def diffusion_coefficients_(self, values: ArrayLike):
        """
        Sets the instance variable _diffusion_coefficients_.
        Updates the instance variables _mu_ and _n_components_.

        Parameters
        ----------
        values : ArrayLike
            diffusion_coefficients.

        Returns
        -------
        None.

        """
        self._diffusion_coefficients_ = np.array(values, copy=True)
        self._update_mu()
        self.n_components = self._mu_.shape[0]

    def _check(self):
        """
        Validate model parameters prior to fitting.



        Returns
        -------
        None.

        """
        super()._check()
        self._mu_ = np.asarray(self._mu_)
        self.n_components = self._mu_.shape[0]

    def _generate_sample_from_state(self, state: int, random_state: int = None) -> ArrayLike:
        """
        Generate a random sample from a given component.

        Parameters
        ----------
        state : int
            Index of the component to condition on.
        random_state : int, optional
             A random number generator instance. If None, the object’s random_state is used.. The default is None.

        Returns
        -------
        ArrayLike
            A random sample from the emission distribution corresponding to a given component.

        """
        random_state = check_random_state(random_state)
        u = random_state.uniform(low=0.0, high=1.0, size=1)
        value = self.cdf_inverse(u, self._mu_[state])
        return (value)

    def _compute_log_likelihood(self, x: ArrayLike) -> ArrayLike:
        """
        Compute per-component log probability under the model.

        Parameters
        ----------
        x : ArrayLike
            Jump distance sequence.

        Returns
        -------
        ArrayLike
            Log probability of each sample in X for each of the model states.

        """
        ll = np.zeros([np.shape(x)[0], self.n_components])
        for state in range(self.n_components):
            ll[:, state] = self._neg_log_likelihood(x, state).T
        return(ll)

    def _neg_log_likelihood(self, x: ArrayLike, state: int) -> ArrayLike:
        """
        Calculate negative log likelihood for a given observation sequence and mobility state.

        Parameters
        ----------
        x : ArrayLike
            Jump distance sequence.
        state : int
            Mobility state.

        Returns
        -------
        ArrayLike
            negative log likelihood.

        """
        yPred = self.pdf(x, self._mu_[state])
        return(1.0 * np.log(yPred))

    def _initialize_sufficient_statistics(self) -> dict:
        """
        Initialize sufficient statistics required for M-step.
        The method is pure, meaning that it doesn’t change the state of the instance. For extensibility computed statistics are stored in a dictionary.

        Returns
        -------
        dict
            Sufficient statistics.

        """
        # nicht negativ selbst schreiben
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return(stats)

    def _accumulate_sufficient_statistics(self, stats: dict, obs: ArrayLike, framelogprob: ArrayLike,
                                          posteriors: ArrayLike, fwdlattice: ArrayLike, bwdlattice: ArrayLike):
        """
        Update sufficient statistics from a given sample.

        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by _initialize_sufficient_statistics().
        obs : ArrayLike
            Jump Distance sequence.
        framelogprob : ArrayLike
            Log-probabilities of each sample under each of the model states.
        posteriors : ArrayLike
            Posterior probabilities of each sample being generated by each of the model states.
        fwdlattice : ArrayLike
            Log-forward and log-backward probabilities.
        bwdlattice : ArrayLike
            Log-forward and log-backward probabilities.

        Returns
        -------
        None.

        """
        super()._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'd' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs**2)

    def _do_mstep(self, stats: dict):
        """
        Perform the M-step of EM algorithm.

        Parameters
        ----------
        stats : dict
            Sufficient statistics updated from all available samples.

        Returns
        -------
        None.

        """
        super()._do_mstep(stats)
        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, None]
        if 'd' in self.params:
            self._mu_ = stats['obs'] / denom

    def _init(self, X: ArrayLike, lengths: ArrayLike = None):
        """
        Initialize model parameters prior to fitting.

        Parameters
        ----------
        X : ArrayLike
            Jumpm distance sequence.
        lengths : ArrayLike, optional
            Lengths of the individual sequences in X.. The default is None.

        Returns
        -------
        None.

        """
        super()._init(X, lengths=lengths)
        
    def fit(self, X: ArrayLike, lengths: ArrayLike):
        """
        Parameterization of the model based on the given jump distance sequence.
        An initialization step is performed before entering the EM algorithm.
        If you want to avoid this step for a subset of the parameters,
        pass proper init_params keyword argument to ErmineHMM constructor.
        By default, the start-, transition-probability and expected diffusion coefficients are optimized.
        If you want to keep a specific model parameter fix,
        pass proper params keyword argument to ErmineHMM constructor.

        Parameters
        ----------
        X : ArrayLike
            Jump distance sequence.
        lengths : ArrayLike
            Lengths of the individual sequences in X.

        Returns
        -------
        None.

        """
        super().fit(X, lengths)
        self._update_diffusion_coefficients()
    
    def _update_diffusion_coefficients(self):
        """
        Updates the diffusion coefficients

        Returns
        -------
        None.

        """
        self._diffusion_coefficients_ = self.mu_/(self.diffusion_degrees_of_freedom_ * self.tau_)
    
    def _update_mu(self):
        """
        Updates the mean squared displacements.

        Returns
        -------
        None.

        """
        self._mu_ = self.diffusion_degrees_of_freedom_ * self.diffusion_coefficients_ * self.tau_ 

    def pdf(self, x: ArrayLike, mu: float) -> ArrayLike:
        """
        Probability density function for observed jump distances.

        Parameters
        ----------
        x : ArrayLike
            Jump distance sequence.
        mu : float
            Mean squared displacement.

        Returns
        -------
        ArrayLike
            Probability to observe the given jump distances.

        """
        p = ((2 * x)/mu) * np.exp((-np.square(x))/mu)
        return (p)

    def cdf(self, x: ArrayLike, mu: float) -> ArrayLike:
        """
        Cumulative density function for observed jump distances.

        Parameters
        ----------
        x : ArrayLike
            Jump distribution sequence.
        mu : float
            Mean squared displacement.

        Returns
        -------
        ArrayLike
            Cumulative probability to observe the given jump distances.

        """
        p = 1 - np.exp((-np.square(x)) / mu)
        return (p)

    def cdf_inverse(self, u: ArrayLike, mu: float) -> ArrayLike:
        """
        Inverse of the cumulative density function for observation probabilities.

        Parameters
        ----------
        u : ArrayLike
            Cumulative probability.
        mu : float
            Mean squared displacement.

        Returns
        -------
        ArrayLike
            Jump distance associated with the given cumulative probability.

        """
        x = np.sqrt(-1.0 * mu * np.log(1.0-u))
        return (x)
    
    def evaluate(self, X: ArrayLike, lengths: ArrayLike = None) -> dict:
        """
        Evaluate the the model quality for a given sample of observed jump distances.

        Parameters
        ----------
        X : ArrayLike
            Jump distance sequence.
        lengths : ArrayLike, optional
             Lengths of the individual sequences in X. The default is None.

        Returns
        -------
        dict
            Evaluation metrics of observation.

        """
        logLikelihood = super().score(X, lengths=None)     
        dof = self.n_components + self.n_components - 1 + np.square(self.n_components) - self.n_components
        instances = np.shape(X)[0]
        bic = dof * np.log(instances) - 2 * logLikelihood
        aic = 2 * dof - 2 * logLikelihood
        aicc = aic + (2*np.square(dof) + 2 * dof)/(instances - dof - 1)
        dictionary = {"classes": [self.n_components],
                      "tracks": [self.n_features],
                      "dof": [dof],
                      "instances": [instances],
                      "log_likelihood": [logLikelihood],
                      "BIC": [bic],
                      "AIC": [aic],
                      "AICc": [aicc]}
        return(dictionary)
