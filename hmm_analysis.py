#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 20:00:12 2021

@author: malkusch
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ermine.models.ErmineHMM import ErmineHMM

def main():
    model = ErmineHMM(n_components=3)
    model.startprob_ = np.array([0.33, 0.33, 0.34])
    model.transmat_ = np.array([[0.8, 0.1, 0.1],
                                [0.1, 0.8, 0.1],
                                [0.1, 0.1, 0.8]])
    model.diffusion_coefficients_ = np.array([[1000.], [30000.], [80000.0]])
    x=np.ndarray([0,1])
    states = np.ndarray([0,1])
    lengths = []
    l = 100
    for i in range(1000):
        X, Z = model.sample(l, random_state= 42+i)
        x = np.concatenate([x,X])
        states = np.concatenate([states, Z.reshape((l,1))])
        lengths.append(l)
    print(np.shape(x))
    print(x)

# =============================================================================
#     data_df = pd.DataFrame(data={'r': x[:, 0],
#                                   'state': states[:, 0]})
#     print(data_df.head())
#     # data_df[data_df['state'] == 0.0]['r'].plot.kde(xlim=[0, 200], color='r')
#     # data_df[data_df['state'] == 1.0]['r'].plot.kde(xlim=[0, 200], color='g')
#     # data_df[data_df['state'] == 2.0]['r'].plot.kde(xlim=[0, 200], color='b')
#     # plt.show()
#     # #print(data_df[data_df['state'] == 0.0].shape)
#     # #print(data_df[data_df['state'] == 1.0].shape)
#     # #print(data_df[data_df['state'] == 2.0].shape)
# 
#     remodel = ErmineHMM(n_components=3, n_iter=100, init_params="",  params="std", tol=1e-5, verbose=True)
#     remodel.startprob_ = np.array([0.05, 0.9, 0.05])
#     remodel.transmat_ = np.array([[0.1, 0.1, 0.8],
#                                   [0.1, 0.8, 0.1],
#                                   [0.1, 0.1, 0.8]])
#     
#     remodel.diffusion_coefficients_ = np.array([[2000.], [50000.], [70000.0]])
#     remodel.fit(x, lengths)
#     print(remodel.startprob_)
#     print(remodel.transmat_)
#     print(remodel.diffusion_coefficients_)
#     
#     results = pd.DataFrame(remodel.evaluate(x, lengths))
#     print(results)
# 
# =============================================================================


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()