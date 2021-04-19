#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 09:23:25 2021

@author: malkusch
"""
import numpy as np
import pandas as pd
import ermine as em
from scipy.stats.distributions import chi2
from matplotlib import pyplot as plt

def main():
    
    file_name = "/Users/malkusch/Documents/Biophysik/ermine/raw_data/inlb/cs4/InlB_CS4_cell01.tracked.csv"
    data_df = pd.read_csv(filepath_or_buffer = file_name)
    jump_df = em.preprocess_swift_data(data_df)
    x_jmm = jump_df["jump_distance"].values
    x_hmm, lengths = em.create_observation_sequence(jump_df)
    
    np.random.seed(42)
    n_components = 2
    jmm = em.JumpDistanceMixtureModel(n_components=n_components)
    jmm.fit(x_jmm)
    
    localization_precision = data_df["uncertainty_xy [nm]"].values.mean()
    tau = 0.02
    sigma = 0.02
    expected_msd = jmm._mu
    diff_coeff = em.postprocessing.error_correction.calculate_diffusion_coefficient(expected_value=expected_msd,
                                                                                    tau=tau,
                                                                                    sigma=sigma,
                                                                                    epsilon=np.sqrt((jmm._mu[0]/4.0)))
    print(localization_precision)
    print(np.sqrt((jmm._mu[0]/4.0)))
    print(jmm.diffusion_coefficients())
    print(diff_coeff)
    

    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()