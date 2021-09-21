#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:54:40 2021

@project: pyErmine
@author: Sebastian Malkusch
@email: malkusch@med.uni-frankfurt.de
@brief: The module contains pre-processing functions for the conversion of the swift data
in order to be able to analyze them afterwards with the Ermine models.
Module functions are:
    preprocess_swift_data
    create_observation_sequence
    init_transition_matrix
"""
from .preprocess_swift_data import preprocess_swift_data
from .create_observation_sequence import create_observation_sequence
from .init_transition_matrix import init_transition_matrix

__all__=["preprocess_swift_data",
         "create_observation_sequence",
         "init_transition_matrix"]