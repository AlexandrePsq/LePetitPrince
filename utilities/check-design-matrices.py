#! usr/bin/env python


""" report various stats for a given design-matrix """
import sys
import os
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import glob

sns.set(style='white')



if __name__ =='__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate password from key (and optionnally the login).""")
    parser.add_argument("--model_name", type=str, help="Name of the model to analysis.")

    args = parser.parse_args()

    input_folder = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/design-matrices/english/{}'.format(args.model_name)
    output_folder = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/design-matrices/english/{}'.format(args.model_name)

    design_matrices = sorted(glob.glob(os.path.join(input_folder, 'design-matrices*')))
    i = 1
    df_final = []

    for dm in design_matrices:
        df = pd.DataFrame({'Feature_name': [], 'Mean':[], 'Std':[], 'Condition_number':[]})
        print('Run #{}'.format(i))
        file_name = dm.split('_')[-1]
        print('\n\n# ' + dm)
        dtxmat = pd.read_csv(dm)
        dtxmat['constant'] = pd.Series(np.ones(dtxmat.shape[0]))
        m = dtxmat.values
        print('\n## Means and STD')
        for c in dtxmat:
            mean = round(dtxmat[c].mean(), 3)
            std = round(dtxmat[c].std(), 3)
            name = "%-15s" % c
            df["Feature_name"] = name
            df["Mean"] = mean
            df["Std"] = std
            print(name, mean, std)
        print('\n## Pairwise Correlations')
        corr = dtxmat.corr()
        print(round(corr, 3))
        print("\n## Condition number", round(np.linalg.cond(m, p=None), 2))
        df["Condition_number"] = round(np.linalg.cond(m, p=None), 2)
        df_final.append(df)
        print("\n## Variance inflation factors:")
        vifs = np.array([round(vif(m, i), 3) for i in range(m.shape[1])])
        vifs = vifs.reshape(vifs.shape[0], 1)
        for i, label in enumerate(dtxmat.columns):
            print("%-15s" % label, vifs[i])
        sns.heatmap(vifs)
        plt.savefig(os.path.join(output_folder, file_name)+'pairwise_correlations.png')
        plt.show()
        plt.close()
        i+=1