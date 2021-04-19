#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:02:50 2021

@author: malkusch
"""
import numpy as np
import pandas as pd
import seaborn as sns
import ermine as em
from matplotlib import pyplot as plt

def main():
    file_name = "/Users/malkusch/Documents/Biophysik/ermine/raw_data/inlb/cs4/InlB_CS4_cell02.tracked.csv"
    data_df = pd.read_csv(filepath_or_buffer = file_name)
    n_components = 3
    np.random.seed(42)
    jump_df = em.preprocess_swift_data(data_df, min_track_length=1)
    x = jump_df["jump_distance"].values
    jmm = em.JumpDistanceMixtureModel(n_components=n_components)
    jmm.fit(x)
    model_df = pd.DataFrame([{"min_track": 1,
                              "w_1": jmm._weights[0],
                              "w_2": jmm._weights[1],
                              "w_3": jmm._weights[2],
                              "D_1": jmm.diffusion_coefficients()[0][0],
                              "D_2": jmm.diffusion_coefficients()[1][0],
                              "D_3": jmm.diffusion_coefficients()[2][0],
                              }])
    for i in range(2,20,1):
        np.random.seed(42)
        jump_df = em.preprocess_swift_data(data_df, min_track_length=i)
        x = jump_df["jump_distance"].values
        jmm = em.JumpDistanceMixtureModel(n_components=n_components)
        jmm.fit(x)
        model_df = model_df.append(pd.DataFrame([{"min_track": i,
                                                  "w_1": jmm._weights[0],
                                                  "w_2": jmm._weights[1],
                                                  "w_3": jmm._weights[2],
                                                  "D_1": jmm.diffusion_coefficients()[0][0],
                                                  "D_2": jmm.diffusion_coefficients()[1][0],
                                                  "D_3": jmm.diffusion_coefficients()[2][0],
                                                  }])
                                   )

    print(model_df)
    sns.lineplot(data=model_df, x="min_track", y="w_1")
    plt.show()
    plt.close()
    sns.lineplot(data=model_df, x="min_track", y="w_2")
    plt.show()
    plt.close()
    sns.lineplot(data=model_df, x="min_track", y="w_3")
    plt.show()
    plt.close()
    sns.lineplot(data=model_df, x="min_track", y="D_1")
    plt.show()
    plt.close()
    sns.lineplot(data=model_df, x="min_track", y="D_2")
    plt.show()
    plt.close()
    sns.lineplot(data=model_df, x="min_track", y="D_3")
    plt.show()
    plt.close()
    


    
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()