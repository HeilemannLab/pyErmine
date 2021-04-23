#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 16:36:21 2021

@author: malkusch
"""

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

def likelihood_ratio_test(ll_min, ll_max, dof_min, dof_max):
    """
    

    Parameters
    ----------
    ll_min : TYPE
        DESCRIPTION.
    ll_max : TYPE
        DESCRIPTION.
    dof_min : TYPE
        DESCRIPTION.
    dof_max : TYPE
        DESCRIPTION.

    Returns
    -------
    lr : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.

    """
    lr = 2 * (ll_max - ll_min)
    delta_dof = dof_max - dof_min
    p = stats.chisqprob(lr,delta_dof)
    return lr, p