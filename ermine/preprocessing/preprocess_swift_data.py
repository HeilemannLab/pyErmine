#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:50:44 2021

@project: pyErmine
@author: Sebastian Malkusch
@email: malkusch@med.uni-frankfurt.de
"""
import numpy as np
import pandas as pd

def preprocess_swift_data(data_df: pd.DataFrame, min_track_length: int = 4) -> pd.DataFrame:
    """
    Creates a data frame of single molecule jumps
    
    The passed in data frame `data_df` is a swift data frame. Here, each line
    represents a single molecule location at a distinct time point.
    Each molecule location in data_df is a departure location of a possible
    molecule jump. For each departure location the respective destination
    location of the same molecule is searched for in the adjacent frame. If
    the respective destnation location is identified, the departure
    and destination location information will define a molecule jump. The
    feature 'jump_distance' characterizes the width of the jump calculated by
    the Euclidean distance between the two localizations. The function returns
    the data frame `jump_df`.
    Essential features of `data_df` are:
        `track.lifetime`
        `track.id`
        `frame`
        `x [nm]`
        `y [nm]`
    

    Parameters
    ----------
    data_df : pd.DataFrame
        A single molecule location data frame created by Swift.
    min_track_length : int, optional
        minimal track length. The default is 4.

    Returns
    -------
    pd.DataFrame
        jump_df:  A single molecule jump data frame.

    """
    filtered_df = data_df[data_df["track.lifetime"] >= min_track_length].copy()
    departure_df = filtered_df.sort_values(by = ["track.id", "frame"], ignore_index=True).copy()
    destination_df = departure_df.drop(0)
    attribute_names = filtered_df.columns
    final_row = pd.Series(data = np.repeat(np.nan, np.shape(attribute_names)[0]),
                          index = attribute_names)
    destination_df = destination_df.append(final_row, ignore_index=True)
    jump_df = departure_df.join(destination_df, on=None, how="left", lsuffix="_departure", rsuffix="_destination")
    jump_df["jump_distance"] = np.sqrt(np.square(jump_df["x [nm]_destination"] - jump_df["x [nm]_departure"]) + np.square(jump_df["y [nm]_destination"] - jump_df["y [nm]_departure"]))
    jump_df["jump_distance"] += np.finfo(np.float32).eps
    trackId_mismatch_idx = jump_df["track.id_departure"] != jump_df["track.id_destination"]
    frame_mismatch_idx = (jump_df["frame_destination"] - jump_df["frame_departure"]) != 1
    jump_df.loc[trackId_mismatch_idx, "jump_distance"] = np.nan
    jump_df.loc[frame_mismatch_idx, "jump_distance"] = np.nan
    return(jump_df.dropna())