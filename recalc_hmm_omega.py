#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:41:38 2021

@author: malkusch
"""
import numpy as np
import pandas as pd
import seaborn as sns
import ermine as em
from matplotlib import pyplot as plt

def define_data_file_path(folder, jmm):
    ligand = jmm["ligand"]
    coverslip = jmm["coverslip"]
    cell = jmm["cell"]
    file_path = str("%s/%s/track_analysis/%s/cells/tracks/%s_%s_%s.tracked.csv" %(folder, ligand, coverslip, ligand, coverslip, cell))
    return(file_path)
                                 
    

def ectract_HMM_init_parameters(jmm):
    # get numer of components
    n_components = jmm["classes"]

    # get pi
    pi = np.zeros(n_components)
    for j in np.arange(np.shape(pi)[0]):
        pi_str = str("omega_%i" %(j+1))
        pi[j] = jmm[pi_str]
    
    # get D
    diff_coef = np.zeros([n_components,1])
    for j in np.arange(np.shape(diff_coef)[0]):
        diff_coef_str = str("apparent_D_%i" %(j+1))
        diff_coef[j] = jmm[diff_coef_str]
    return n_components, pi, diff_coef

data_folder_path = "/home/malkusch/data/cMet/Fab_InlB_diff_limit_14"
file_path = "/home/malkusch/PowerFolders/Met-HMM/code/results/jmm_based_model_selection_210422.csv"
init_model = pd.read_csv(filepath_or_buffer=file_path).iloc[2]
n_components, init_pi, init_diff_coef = ectract_HMM_init_parameters(init_model)
init_trans_mat = em.init_transition_matrix(n_components = n_components, stability = 0.9)

data_file_path = define_data_file_path(folder = data_folder_path, jmm = init_model)
data_df = pd.read_csv(filepath_or_buffer = data_file_path)
jump_df = em.preprocess_swift_data(data_df, min_track_length = 4)
x_jmm = jump_df["jump_distance"].values
x_hmm, lengths = em.create_observation_sequence(jump_df)
    
jump_df.shape[0]

hmm = em.ErmineHMM(n_components = n_components, init_params = "", params="stm", n_iter = 1000, tol = 1e-5)
hmm.startprob_ = init_pi
hmm.transmat_ = init_trans_mat
hmm.diffusion_coefficients_ = init_diff_coef

hmm.fit(x_hmm, lengths)
hmm.diffusion_coefficients_

tau = 0.02
dof = 4
jmm = em.JumpDistanceMixtureModel(n_components=n_components, init_params = "w", params="w")
jmm._mu = np.squeeze(dof * hmm.diffusion_coefficients_ * tau, axis= 1)
jmm.fit(x_jmm)
jmm._weights



model_df = pd.DataFrame({"r": np.arange(0,300,1),
                        "superposition": np.zeros(300)})
for i in np.arange(0, n_components, 1):
    unimodal_judi_model = em.JumpDistanceModel(diffusion_coefficient = jmm.diffusion_coefficients()[i],
                                               degrees_of_freedom = dof,
                                               tau=tau)
    component_string = str("state_%i" %(i+1))
    model_df[component_string] = jmm._weights[i] * unimodal_judi_model.pdf(distance = model_df["r"])
    model_df["superposition"] = model_df["superposition"] + model_df[component_string]

sns.kdeplot(data=jump_df, x="jump_distance", fill = True, bw_adjust = 0.3, clip = [0, 300])
sns.lineplot(data=model_df.melt(id_vars=['r']), x="r", y="value", color="black", style="variable")
plt.show()