# -*- coding: utf-8 -*-

import os
from os.path import join
import argparse

import warnings
warnings.simplefilter(action='ignore')

import numpy as np


def check_folder(path):
    # Create adequate folders if necessary
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate r2 maps from design matrices and fMRI data in a given language for a given model.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--nb_features", type=str, default=None, help="Number of features.")
    parser.add_argument("--output", type=str, default='', help="Path to the file where to save the shuffling array.")
    parser.add_argument("--n_permutations", type=str, default=None, help="Number of permutations.")

    args = parser.parse_args()
    
    n_permutations = int(args.n_permutations)
    np.random.seed(1111)

    columns_index = np.arange(int(args.nb_features))
    shuffling = []

    # computing permutations
    for _ in range(n_permutations):
        np.random.shuffle(columns_index)
        shuffling.append(columns_index.copy())
    np.save(args.output, shuffling)
    #np.save(os.path.join(args.output, 'shuffling.npy'), shuffling)
    