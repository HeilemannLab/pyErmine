#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 14:53:22 2021

@author: malkusch
"""
import numpy as np
import pandas as pd
import seaborn as sns
import ermine as em
from matplotlib import pyplot as plt

def plot_model(jump_df, optModel_components, optModel_diffusion_coefficients, optModel_weights, min_track_length):   
    model_df = pd.DataFrame({"r": np.arange(0,300,1),
                             "superposition": np.zeros(300)})
    for i in np.arange(0, optModel_components, 1):
        unimodal_judi_model = em.JumpDistanceModel(diffusion_coefficient = optModel_diffusion_coefficients[i],
                                                   degrees_of_freedom = 4,
                                                   tau=0.02)
        component_string = str("state_%i" %(i+1))
        model_df[component_string] = optModel_weights[i] * unimodal_judi_model.pdf(distance = model_df["r"])
        model_df["superposition"] = model_df["superposition"] + model_df[component_string]
    
    sns.kdeplot(data=jump_df, x="jump_distance", fill = True, bw_adjust = 0.3, clip = [0, 300])
    sns.lineplot(data=model_df.melt(id_vars=['r']), x="r", y="value", color="black", style="variable")
    plt.title(str("Min Track Length %i" %(min_track_length)))
    plt.show()

def main():
    file_name = "/Users/malkusch/Documents/Biophysik/ermine/raw_data/inlb/cs4/InlB_CS4_cell02.tracked.csv"
    data_df = pd.read_csv(filepath_or_buffer = file_name)
    n_components = 2
    np.random.seed(42)
    jump_df = em.preprocess_swift_data(data_df, min_track_length=1)
    x = jump_df["jump_distance"].values
    jmm = em.JumpDistanceMixtureModel(n_components=n_components)
    jmm.fit(x)
    model_df = pd.DataFrame(jmm.evaluate(x))
    plot_model(jump_df = jump_df,
               optModel_components= n_components,
               optModel_diffusion_coefficients=jmm.diffusion_coefficients(),
               optModel_weights=jmm._weights,
               min_track_length=1)
    for i in range(2,15,1):
        np.random.seed(42)
        jump_df = em.preprocess_swift_data(data_df, min_track_length=i)
        x = jump_df["jump_distance"].values
        jmm = em.JumpDistanceMixtureModel(n_components=n_components)
        jmm.fit(x)
        model_df = model_df.append(pd.DataFrame(jmm.evaluate(x)))
        plot_model(jump_df = jump_df,
               optModel_components= n_components,
               optModel_diffusion_coefficients=jmm.diffusion_coefficients(),
               optModel_weights=jmm._weights,
               min_track_length = i)
        
    model_df["min_track_length"] = np.arange(1,15,1)
    print(model_df)
    fig, ax = plt.subplots()
    model_df.plot.line(x="min_track_length", y="BIC")
    plt.show()
    
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()