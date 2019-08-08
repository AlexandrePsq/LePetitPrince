# -*- coding: utf-8 -*-

import os
from os.path import join
import argparse

import warnings
warnings.simplefilter(action='ignore')

import numpy as np



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate r2 maps from design matrices and fMRI data in a given language for a given model.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--nb_features", type=str, default=None, help="Number of features.")
    parser.add_argument("--shuffling", type=str, default='', help="Path for saving shuffling array.")
    parser.add_argument("--n_permutations", type=str, default=None, help="Number of permutations.")

    args = parser.parse_args()
    
    n_permutations = int(args.n_permutations)

    columns_index = np.arange(int(args.nb_features))
    shuffling = []

    # computing permutations
    for _ in range(n_permutations):
        np.random.shuffle(columns_index)
        shuffling.append(columns_index)
    np.save(args.shuffling, shuffling)
    