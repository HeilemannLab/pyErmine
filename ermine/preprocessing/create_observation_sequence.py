#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 08:25:26 2021

@project: pyErmine
@author: Sebastian Malkusch
@email: malkusch@med.uni-frankfurt.de
"""
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

def create_observation_sequence(judi_df: pd.DataFrame) -> (ArrayLike, ArrayLike):
    """
    Creates an observation sequence of single particle tracking trajectories for the analysis with ErmineHMM.

    Parameters
    ----------
    judi_df : pd.DataFrame
        Pandas DataFrame object that comprises information on "jump_distance" and "track.id_departure".

    Returns
    -------
    (ArrayLike, ArrayLike)
        x: Jump distance observation sequence.
        lengths: Lengths of the individual sequences in x.

    """
    unique_track_id_vec = np.unique(judi_df["track.id_departure"].values)
    track_number = np.shape(unique_track_id_vec)[0]
    lengths = []
    x=np.ndarray([0,1])
    judi_vec = judi_df["jump_distance"].values
    track_id_vec = judi_df["track.id_departure"].values
    for i in np.arange(0,track_number,1):
        idx = track_id_vec == unique_track_id_vec[i]
        track_x = np.expand_dims(judi_vec[idx], axis = 0)
        lengths.append(np.shape(track_x)[1])
        x = np.concatenate([x,track_x.T])
    return(x, lengths)
