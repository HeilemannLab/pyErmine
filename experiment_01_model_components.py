#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 09:53:20 2021

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
    
    np.random.seed(42)

    model_df = pd.DataFrame({"ligand": [],
                             "coverslip": [],
                             "cell": [],
                             "classes": [],
                             "dof": [],
                             "instances": [],
                             "log_likelihood": [],
                             "BIC": [],
                             "AIC": [],
                             "AICc": []})
    
    model_df = model_df.astype({"ligand": 'str',
                    "coverslip": 'str',
                    "cell": 'str',
                    "classes": 'int32',
                    "dof": 'int32',
                    "instances": 'int64',
                    "log_likelihood": 'float',
                    "BIC": 'float',
                    "AIC": 'float',
                    "AICc": 'float'})

    for filename in filenames:
        ligand, coverslip, cell = extract_metadata(filename)
        data_df = pd.read_csv(filepath_or_buffer = filename)
        jump_df = em.preprocess_swift_data(data_df)
        x = jump_df["jump_distance"].values
        for i in np.arange(start=1, stop=7, step=1):
            jmm = em.JumpDistanceMixtureModel(n_components=i)
            jmm.fit(x)
            temp_df = pd.DataFrame(jmm.evaluate(x))
            temp_df["ligand"] = ligand
            temp_df["coverslip"] = coverslip
            temp_df["cell"] = cell
            model_df = model_df.append(temp_df)
        
    model_df.to_csv(path_or_buf = "/Users/malkusch/PowerFolders/Met-HMM/code/results/jmm_based_model_selection.csv") 
    
    
        
   
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()