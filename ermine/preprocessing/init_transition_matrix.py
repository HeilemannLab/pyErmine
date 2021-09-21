#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 11:48:28 2021

@project: pyErmine
@author: Sebastian Malkusch
@email: malkusch@med.uni-frankfurt.de
"""
import numpy as np
from numpy.typing import ArrayLike
type(np.zeros([2,2]))

def init_transition_matrix(n_components: int = 2, stability: float = 0.9) -> ArrayLike:
    """
    Initializes a transition matrix for ermine

    Based on the given parameters `n_components` and `stability` an initial
    guess for a transition matrix is calculated.
    

    Parameters
    ----------
    n_components : int, optional
        Number of model states. The default is 2.
    stability : float, optional
        Decay probability of a state. Must be less than or equal to 1. The default is 0.9.

    Returns
    -------
    ArrayLike
        trans-mat: A n x n trnasition matrix.

    """
    if stability > 1:
        print("warning: stability < 1 is prerequisite. stability will be set to 0.9")
        stability = 0.9
    trans_mat = np.ones([n_components, n_components])
    if(n_components>1):
        for i in np.arange(0,n_components,1):
            for j in np.arange(0,n_components,1):
                if i != j:
                    trans_mat[i,j] = (1.0 - stability)/(n_components - 1)
                else:
                    trans_mat[i,j] = stability
    return(trans_mat)
        