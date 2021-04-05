#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:05:53 2021

@author: malkusch
"""
import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2
from matplotlib import pyplot as plt

from ermine.preprocessing.preprocess_swift_data import preprocess_swift_data
from ermine.models.JumpDistanceMixtureModel import JumpDistanceMixtureModel


def likelihood_ratio(llmin, llmax):
    return(2*(llmax - llmin))
    
def main():
    file_name = "/Users/malkusch/Documents/Biophysik/ermine/raw_data/inlb/cs4/InlB_CS4_cell01.tracked.csv"
    data_df = pd.read_csv(filepath_or_buffer = file_name)
    jump_df = preprocess_swift_data(data_df)
    x = jump_df["jump_distance"].values
    
    np.random.seed(47)
    jmm = JumpDistanceMixtureModel(n_components=1)
    jmm.fit(x)
    model_df = pd.DataFrame(jmm.evaluate(x))
    for i in np.arange(start=2, stop=6, step=1):
        jmm = JumpDistanceMixtureModel(n_components=i)
        jmm.fit(x)
        model_df = model_df.append(pd.DataFrame(jmm.evaluate(x)))
        
    model_df.plot.line(x="classes", y="BIC")
    plt.show()
    
    print(model_df)
    
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
    
    jmm = JumpDistanceMixtureModel(n_components=2)
    jmm.fit(x)
    print(jmm.diffusion_coefficients())
    print(jmm._weights)
    
    jump_df["jump_distance"].plot.kde(xlim = [-10, 400])
    plt.show()
    
    
    
        
   
    


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()