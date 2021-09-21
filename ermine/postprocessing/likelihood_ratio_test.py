#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:36:21 2021

@project: pyErmine
@author: Sebastian Malkusch
@email: malkusch@med.uni-frankfurt.de
"""

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

def likelihood_ratio_test(ll_min: float, ll_max: float, dof_min: int, dof_max: int) -> (float, float):
    """
    Assesses the goodness of fit of two competing statistical models based on the ratio of their likelihoods.
    

    Parameters
    ----------
    ll_min : float
        Likelihood of the less complex model.
    ll_max : float
        Likelihood of the more complex model.
    dof_min : int
        Degrees of freedom of the less complex model.
    dof_max : int
        Degrees of freedom of the more complex model.

    Returns
    -------
    (float, float)
        lr: Likelihood ratio.
        p: p Value.

    """
    lr = 2 * (ll_max - ll_min)
    delta_dof = dof_max - dof_min
    p = stats.chisqprob(lr,delta_dof)
    return (lr, p)