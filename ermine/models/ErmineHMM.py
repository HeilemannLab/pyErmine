# from hmmlearn import hmm
from hmmlearn.base import _BaseHMM

import numpy as np
from sklearn.utils import check_random_state

class ErmineHMM(_BaseHMM):
    r"""Hidden Markov Model with Gaussian emissions.
    Parameters
    ----------
    n_components : int
        Number of states.

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.

    Examples
    --------
    >>> from hmmlearn.hmm import GaussianHMM
    >>> GaussianHMM(n_components=2)  #doctest: +ELLIPSIS
    GaussianHMM(algorithm='viterbi',...
    """

    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 diffusion_prior=0, diffusion_weight=0,
                 diffusion_degrees_of_freedom = 4,
                 tau = 0.02,
                 algorithm="viterbi", random_state=42,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="std", init_params="std"):
        _BaseHMM.__init__(self, n_components=n_components, startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose, init_params=init_params)

        self._tau_ = tau
        # self.diffusion_prior_ = diffusion_prior
        # self.diffusion_weight_ = diffusion_weight
        self._diffusion_degrees_of_freedom_ = diffusion_degrees_of_freedom
        self._diffusion_coefficients_ = np.zeros([self.n_components])
        self._mu_ = np.zeros([self.n_components])
        self.n_features = 1


    @property
    def tau_(self):
        """Return instance variable tau."""
        return (self._tau_)

    @tau_.setter
    def tau_(self, value):
        """Sets instance variable tau."""
        self._tau_ = float(value)
        
    @property
    def diffusion_degrees_of_freedom_ (self):
        """Return instance variable diffusion_degrees_of_freedom_."""
        return (self._diffusion_degrees_of_freedom_ )

    @diffusion_degrees_of_freedom_.setter
    def diffusion_degrees_of_freedom_ (self, value):
        """Sets instance variable tau."""
        self._diffusion_degrees_of_freedom_ = float(value)

    @property
    def diffusion_prior_(self):
        """Return diffusion_coefficients."""
        return (self._diffusion_prior_)

    @diffusion_prior_.setter
    def diffusion_prior_(self, value):
        self._diffusion_prior_ = np.float(value)

    @property
    def diffusion_weight_(self):
        """Return diffusion_coefficients."""
        return (self._diffusion_weight_)

    @diffusion_weight_.setter
    def diffusion_weight_(self, value):
        self._diffusion_weight_ = np.float(value)
        
    @property
    def mu_(self):
        """Return expectation values mu."""
        return (self._mu_)
    
    @mu_.setter
    def mu_(self, values):
        self._mu_ = np.array(values, copy=True)
        self._update_diffusion_coefficients()
        self.n_components = self._mu_.shape[0]
        
    @property
    def diffusion_coefficients_(self):
        """Return expectation values diffusion_coefficients."""
        return (self._diffusion_coefficients_)
    
    @diffusion_coefficients_.setter
    def diffusion_coefficients_(self, values):
        self._diffusion_coefficients_ = np.array(values, copy=True)
        self._update_mu()
        self.n_components = self._mu_.shape[0]

    def _check(self):
        super()._check()
        self._mu_ = np.asarray(self._mu_)
        self.n_components = self._mu_.shape[0]

    def _generate_sample_from_state(self, state, random_state=None):
        random_state = check_random_state(random_state)
        u = random_state.uniform(low=0.0, high=1.0, size=1)
        value = self.cdf_inverse(u, self._mu_[state])
        return (value)

    def _compute_log_likelihood(self, x):
        ll = np.zeros([np.shape(x)[0], self.n_components])
        for state in range(self.n_components):
            ll[:, state] = self._neg_log_likelihood(x, state).T
        return(ll)

    def _neg_log_likelihood(self, x, state):
        yPred = self.pdf(x, self._mu_[state])
        return(1.0 * np.log(yPred))

    def _initialize_sufficient_statistics(self):
        # nicht negativ selbst schreiben
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return(stats)

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'd' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs**2)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, None]
        if 'd' in self.params:
            self._mu_ = stats['obs'] / denom

    def _init(self, X, lengths=None):
        super()._init(X, lengths=lengths)
        
    def fit(self, X, lengths):
        super().fit(X, lengths)
        self._update_diffusion_coefficients()

    # def _check(self):
    #     super()._check()
    #     self.diffusion_coefficients_ = np.asarray(self.diffusion_coefficients_)
    #     self.n_features = 1
    
    def _update_diffusion_coefficients(self):
        self._diffusion_coefficients_ = self.mu_/(self.diffusion_degrees_of_freedom_ * self.tau_)
    
    def _update_mu(self):
        self._mu_ = self.diffusion_degrees_of_freedom_ * self.diffusion_coefficients_ * self.tau_ 

    def pdf(self, x, mu):
        p = ((2 * x)/mu) * np.exp((-np.square(x))/mu)
        return (p)

    def cdf(self, x, mu):
        p = 1 - np.exp((-np.square(x)) / mu)
        return (p)

    def cdf_inverse(self, u, mu):
        x = np.sqrt(-1.0 * mu * np.log(1.0-u))
        return (x)
    
    def evaluate(self, X, lengths=None):
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
