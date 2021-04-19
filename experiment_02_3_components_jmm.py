#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:36:54 2021

@author: malkusch
"""
import numpy as np
import pandas as pd
import ermine as em
import os


def query_filenames(path):
    filenames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if(file.endswith(".tracked.csv")):
                filenames.append(os.path.join(root,file))
    return(filenames)

def extract_metadata(filename):
    basename = os.path.basename(filename)
    experiment = basename.split(".")[0]
    ligand, coverslip, cell = experiment.split("_")
    return(ligand, coverslip, cell)

def main():
    path = "/Users/malkusch/PowerFolders/Met-HMM/code/data"
    filenames =  query_filenames(path)
    
    model_df = pd.DataFrame({"ligand": [],
                             "coverslip": [],
                             "cell": [],
                             "classes": [],
                             "dof": [],
                             "instances": [],
                             "log_likelihood": [],
                             "BIC": [],
                             "AIC": [],
                             "AICc": [],
                             "omega_1": [],
                             "omega_2": [],
                             "omega_3": [],
                             "omega_4": [],
                             "omega_5": [],
                             "omega_6": [],
                             "apparent_D_1": [],
                             "apparent_D_2": [],
                             "apparent_D_3": [],
                             "apparent_D_4": [],
                             "apparent_D_5": [],
                             "apparent_D_6": []})
    
    
    for filename in filenames:       
        ligand, coverslip, cell = extract_metadata(filename)
        data_df = pd.read_csv(filepath_or_buffer = filename)
        jump_df = em.preprocess_swift_data(data_df, min_track_length = 4)
        x = jump_df["jump_distance"].values
        for n_components in np.arange(start=1, stop=7, step=1):
            np.random.seed(42)
            jmm = em.JumpDistanceMixtureModel(n_components=n_components)
            jmm.fit(x)
            
            diff_coeff = jmm.diffusion_coefficients()
            temp_df = pd.DataFrame(jmm.evaluate(x))
            temp_df["ligand"] = ligand
            temp_df["coverslip"] = coverslip
            temp_df["cell"] = cell
            temp_df["omega_1"] = [jmm._weights[0]]
            temp_df["apparent_D_1"] = diff_coeff[0]
            if(n_components > 1):
                temp_df["omega_2"] = [jmm._weights[1]]
                temp_df["apparent_D_2"] = diff_coeff[1]
            else:
                temp_df["omega_2"] = [np.nan]
                temp_df["apparent_D_2"] = [np.nan]
            if(n_components > 2):
                temp_df["omega_3"] = [jmm._weights[2]]
                temp_df["apparent_D_3"] = diff_coeff[2]
            else:
                temp_df["omega_3"] = [np.nan]
                temp_df["apparent_D_3"] = [np.nan]
            if(n_components > 3):
                temp_df["omega_4"] = [jmm._weights[3]]
                temp_df["apparent_D_4"] = diff_coeff[3]
            else:
                temp_df["omega_4"] = [np.nan]
                temp_df["apparent_D_4"] = [np.nan]
            if(n_components > 4):
                temp_df["omega_5"] = [jmm._weights[4]]
                temp_df["apparent_D_5"] = diff_coeff[4]
            else:
                temp_df["omega_5"] = [np.nan]
                temp_df["apparent_D_5"] = [np.nan]     
            if(n_components > 5):
                temp_df["omega_6"] = [jmm._weights[5]]
                temp_df["apparent_D_6"] = diff_coeff[5]
            else:
                temp_df["omega_6"] = [np.nan]
                temp_df["apparent_D_6"] = [np.nan] 
            
            model_df = model_df.append(temp_df)
    
    model_df = model_df = model_df.astype({"ligand": "str",
                                           "coverslip": "str",
                                           "cell": "str",
                                           "classes": "int32",
                                           "dof": "int32",
                                           "instances": "int64",
                                           "log_likelihood": "float",
                                           "BIC": "float",
                                           "AIC": "float",
                                           "AICc": "float",
                                           "omega_1": "float",
                                           "omega_2": "float",
                                           "omega_3": "float",
                                           "omega_4": "float",
                                           "omega_5": "float",
                                           "omega_6": "float",
                                           "apparent_D_1": "float",
                                           "apparent_D_2": "float",
                                           "apparent_D_3": "float",
                                           "apparent_D_4": "float",
                                           "apparent_D_5": "float",
                                           "apparent_D_6": "float"})
    
        
    model_df.to_csv(path_or_buf = "/Users/malkusch/PowerFolders/Met-HMM/code/results/jmm_based_model_selection.csv") 
    print(model_df.head())
    
    
        
   
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()