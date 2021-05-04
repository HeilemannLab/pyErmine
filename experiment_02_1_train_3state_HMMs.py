#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 09:07:27 2021

@author: malkusch
"""


import numpy as np
import pandas as pd
import ermine as em

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

def main():
    # init_model_file_path = "/Users/malkusch/PowerFolders/Met-HMM/code/data/jmm_based_model_selection_210422.csv"
    init_model_file_path = "/home/malkusch/PowerFolders/Met-HMM/code/results/jmm_based_model_selection_210422.csv"
    # data_folder_path = "/Users/malkusch/Documents/Biophysik/ermine/raw_data"
    data_folder_path = "/home/malkusch/data/cMet/Fab_InlB_diff_limit_14"
    init_model_df = pd.read_csv(filepath_or_buffer=init_model_file_path)
    init_model_df = init_model_df[init_model_df["classes"] == 3]
    
    tau = 0.02
    dof = 4
    max_n_components = 6
    model_df = pd.DataFrame({"ligand": [],
                             "coverslip": [],
                              "cell": [],
                              "classes": [],
                              "tracks": [],
                              "dof": [],
                              "instances": [],
                              "log_likelihood": [],
                              "BIC": [],
                              "AIC": [],
                              "AICc": [],
                              "pi_1": [],
                              "pi_2": [],
                              "pi_3": [],
                              "pi_4": [],
                              "pi_5": [],
                              "pi_6": [],
                              "apparent_D_1": [],
                              "apparent_D_2": [],
                              "apparent_D_3": [],
                              "apparent_D_4": [],
                              "apparent_D_5": [],
                              "apparent_D_6": [],
                              "P(1|1)": [],
                              "P(2|1)": [],
                              "P(3|1)": [],
                              "P(4|1)": [],
                              "P(5|1)": [],
                              "P(6|1)": [],
                              "P(1|2)": [],
                              "P(2|2)": [],
                              "P(3|2)": [],
                              "P(4|2)": [],
                              "P(5|2)": [],
                              "P(6|2)": [],
                              "P(1|3)": [],
                              "P(2|3)": [],
                              "P(3|3)": [],
                              "P(4|3)": [],
                              "P(5|3)": [],
                              "P(6|3)": [],
                              "P(1|4)": [],
                              "P(2|4)": [],
                              "P(3|4)": [],
                              "P(4|4)": [],
                              "P(5|4)": [],
                              "P(6|4)": [],
                              "P(1|5)": [],
                              "P(2|5)": [],
                              "P(3|5)": [],
                              "P(4|5)": [],
                              "P(5|5)": [],
                              "P(6|5)": [],
                              "P(1|6)": [],
                              "P(2|6)": [],
                              "P(3|6)": [],
                              "P(4|6)": [],
                              "P(5|6)": [],
                              "P(6|6)": [],
                              "omega_1": [],
                              "omega_2": [],
                              "omega_3": [],
                              "omega_4": [],
                              "omega_5": [],
                              "omega_6": [],
                              "epsilon": []})
    
    for i in np.arange(init_model_df.shape[0]):
    # for i in np.arange(3):
        np.random.seed(42)
        recent_model = init_model_df.iloc[i]
        n_components, init_pi, init_diff_coef = ectract_HMM_init_parameters(recent_model)
        init_trans_mat = em.init_transition_matrix(n_components = n_components, stability = 0.9)
        
        hmm = em.ErmineHMM(n_components = n_components,
                           diffusion_degrees_of_freedom = dof,
                           tau = tau,
                           init_params = "",
                           params="stm",
                           n_iter = 1000,
                           tol = 1e-5)
        hmm.startprob_ = init_pi
        hmm.transmat_ = init_trans_mat
        hmm.diffusion_coefficients_ = init_diff_coef
        
        data_file_path = define_data_file_path(folder = data_folder_path, jmm = recent_model)
    
        
        data_df = pd.read_csv(filepath_or_buffer = data_file_path)
        jump_df = em.preprocess_swift_data(data_df, min_track_length = 4)
        x_jmm = jump_df["jump_distance"].values
        x_hmm, lengths = em.create_observation_sequence(jump_df)
        
        hmm.fit(x_hmm, lengths)
        
        pi = np.zeros(max_n_components) * np.nan
        pi[:n_components] = hmm.startprob_
        
        diff_coef = np.zeros(max_n_components) * np.nan
        diff_coef[:n_components] = hmm.diffusion_coefficients_[:,0]
        
        trans_mat =  np.zeros([max_n_components, max_n_components]) * np.nan
        trans_mat[:n_components, :n_components] = hmm.transmat_
        
        jmm = em.JumpDistanceMixtureModel(n_components=n_components,
                                          degrees_of_freedom = dof,
                                          tau=tau,
                                          init_params = "w",
                                          params="w")
        jmm._mu = np.squeeze(dof * hmm.diffusion_coefficients_ * tau, axis= 1)
        jmm.fit(x_jmm)
        omega = np.zeros(max_n_components) * np.nan
        omega[:n_components] = jmm._weights
        
        recent_model_df = pd.DataFrame(hmm.evaluate(x_hmm, lengths))
        recent_model_df["ligand"] = recent_model["ligand"]
        recent_model_df["coverslip"] = recent_model["coverslip"]
        recent_model_df["cell"] = recent_model["cell"]
        for k in np.arange(max_n_components):
            pi_str = str("pi_%i" %(k+1))
            recent_model_df[pi_str] = pi[k]
            diff_coef_str = str("apparent_D_%i" %(k+1))
            recent_model_df[diff_coef_str] = diff_coef[k]
            for l in np.arange(max_n_components):
                trans_mat_str = str("P(%i|%i)" %(l+1, k+1))
                recent_model_df[trans_mat_str] = trans_mat[l,k]
            omega_str = str("omega_%i" %(k+1))
            recent_model_df[omega_str] = omega[k]
        recent_model_df["epsilon"] = recent_model["epsilon"]
        
        model_df = model_df.append(recent_model_df)
    model_df.columns
    model_df = model_df.astype({"ligand": "str",
                                "coverslip": "str",
                                "cell": "str",
                                "classes": "int32",
                                "tracks": "int32",
                                "dof": "int32",
                                "instances": "int32",
                                "log_likelihood": "float",
                                "BIC": "float",
                                "AIC": "float",
                                "AICc": "float",
                                "pi_1": "float",
                                "pi_2": "float",
                                "pi_3": "float",
                                "pi_4": "float",
                                "pi_5": "float",
                                "pi_6": "float",
                                "apparent_D_1": "float",
                                "apparent_D_2": "float",
                                "apparent_D_3": "float",
                                "apparent_D_4": "float",
                                "apparent_D_5": "float",
                                "apparent_D_6": "float",
                                "P(1|1)": "float",
                                "P(2|1)": "float",
                                "P(3|1)": "float",
                                "P(4|1)": "float",
                                "P(5|1)": "float",
                                "P(6|1)": "float",
                                "P(1|2)": "float",
                                "P(2|2)": "float",
                                "P(3|2)": "float",
                                "P(4|2)": "float",
                                "P(5|2)": "float",
                                "P(6|2)": "float",
                                "P(1|3)": "float",
                                "P(2|3)": "float",
                                "P(3|3)": "float",
                                "P(4|3)": "float",
                                "P(5|3)": "float",
                                "P(6|3)": "float",
                                "P(1|4)": "float",
                                "P(2|4)": "float",
                                "P(3|4)": "float",
                                "P(4|4)": "float",
                                "P(5|4)": "float",
                                "P(6|4)": "float",
                                "P(1|5)": "float",
                                "P(2|5)": "float",
                                "P(3|5)": "float",
                                "P(4|5)": "float",
                                "P(5|5)": "float",
                                "P(6|5)": "float",
                                "P(1|6)": "float",
                                "P(2|6)": "float",
                                "P(3|6)": "float",
                                "P(4|6)": "float",
                                "P(5|6)": "float",
                                "P(6|6)": "float",
                                "omega_1": "float",
                                "omega_2": "float",
                                "omega_3": "float",
                                "omega_4": "float",
                                "omega_5": "float",
                                "omega_6": "float",
                                "epsilon": "float"})
    
    # model_df.to_csv(path_or_buf = "/Users/malkusch/PowerFolders/Met-HMM/code/results/hmm_based_model_selection.csv")
    model_df.to_csv(path_or_buf = "/home/malkusch/PowerFolders/Met-HMM/code/results/hmm_based_3-state-models.csv") 
    print(model_df.head())

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()