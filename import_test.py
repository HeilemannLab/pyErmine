#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 08:08:58 2021

@author: malkusch
"""
import numpy as np
import pandas as pd
import ermine as em
from scipy.stats.distributions import chi2
from matplotlib import pyplot as plt

def likelihood_ratio(llmin, llmax):
    return(2*(llmax - llmin))

def main():
    
    file_name = "/Users/malkusch/Documents/Biophysik/ermine/raw_data/inlb/cs4/InlB_CS4_cell01.tracked.csv"
    data_df = pd.read_csv(filepath_or_buffer = file_name)
    jump_df = em.preprocess_swift_data(data_df)
    x_jmm = jump_df["jump_distance"].values
    x_hmm, lengths = em.create_observation_sequence(jump_df)
    
    np.random.seed(42)
    n_components = 1
    jmm = em.JumpDistanceMixtureModel(n_components=n_components)
    jmm.fit(x_jmm)

    
    hmm = em.ErmineHMM(n_components=n_components, n_iter=100, init_params="",  params="std", tol=1e-4, verbose=False)    
    hmm.startprob_ = jmm._weights
    hmm.diffusion_coefficients_ = jmm.diffusion_coefficients()
    hmm.transmat_ = em.init_transition_matrix(n_components=n_components, stability=0.8)
    hmm.fit(x_hmm, lengths)
    model_df = pd.DataFrame(hmm.evaluate(x_hmm, lengths))
    
    for i in np.arange(2,7,1):
        print(i)
        n_components = i
        jmm = em.JumpDistanceMixtureModel(n_components=n_components)
        jmm.fit(x_jmm)

    
        hmm = em.ErmineHMM(n_components=n_components, n_iter=100, init_params="",  params="std", tol=1e-4, verbose=False)    
        hmm.startprob_ = jmm._weights
        hmm.diffusion_coefficients_ = jmm.diffusion_coefficients()
        hmm.transmat_ = em.init_transition_matrix(n_components=n_components, stability=0.8)
        hmm.fit(x_hmm, lengths)
        model_df = model_df.append(pd.DataFrame(hmm.evaluate(x_hmm, lengths)))
    
    llmin = model_df["log_likelihood"].values[0]
    dof_min = model_df["dof"].values[0]
    for i in np.arange(start=1, stop=5, step=1):
        llmax = model_df["log_likelihood"].values[i]
        dof_max = model_df["dof"].values[i]
        lr = likelihood_ratio(llmin, llmax)
        delta_dof = dof_max - dof_min
        p = chi2.sf(lr, delta_dof)
        print(i+1)
        print(p)
        llmin = llmax
        dof_min = dof_max
    
    print(model_df)
    
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    
# =============================================================================
#     
# c = np.array([[0.1, 0.1, 0.8],
#               [0.1, 0.8, 0.1],
#               [0.1, 0.1, 0.8]])
# print(np.shape(c))
# c_1 = np.zeros()
# =============================================================================
