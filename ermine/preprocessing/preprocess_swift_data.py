#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:50:44 2021

@author: malkusch
"""
import numpy as np
import pandas as pd

def preprocess_swift_data(data_df, min_track_length = 4):
    filtered_df = data_df[data_df["track.lifetime"] >= min_track_length].copy()
    departure_df = filtered_df.sort_values(by = ["track.id", "frame"], ignore_index=True).copy()
    destination_df = departure_df.drop(0)
    attribute_names = filtered_df.columns
    final_row = pd.Series(data = np.repeat(np.nan, np.shape(attribute_names)[0]),
                          index = attribute_names)
    destination_df = destination_df.append(final_row, ignore_index=True)
    jump_df = departure_df.join(destination_df, on=None, how="left", lsuffix="_departure", rsuffix="_destination")
    jump_df["jump_distance"] = np.sqrt(np.square(jump_df["x [nm]_destination"] - jump_df["x [nm]_departure"]) + np.square(jump_df["y [nm]_destination"] - jump_df["y [nm]_departure"]))
    trackId_mismatch_idx = jump_df["track.id_departure"] != jump_df["track.id_destination"]
    frame_mismatch_idx = (jump_df["frame_destination"] - jump_df["frame_departure"]) != 1
    jump_df.loc[trackId_mismatch_idx, "jump_distance"] = np.nan
    jump_df.loc[frame_mismatch_idx, "jump_distance"] = np.nan
    return(jump_df.dropna())