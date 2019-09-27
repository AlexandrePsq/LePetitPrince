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
        
        print('Run #{}'.format(i))
        file_name = dm.split('_')[-1]
        print('\n\n# ' + dm)
        dtxmat = pd.read_csv(dm)
        dtxmat['constant'] = pd.Series(np.ones(dtxmat.shape[0]))
        m = dtxmat.values
        print('\n## Means and STD')
        name = []
        mean = []
        std =[]
        for c in dtxmat:
            mean.append(round(dtxmat[c].mean(), 3))
            std.append(round(dtxmat[c].std(), 3))
            name.append("%-15s" % c)
            print(name, mean, std)
        print('\n## Pairwise Correlations')
        corr = dtxmat.corr()
        print(round(corr, 3))
        print("\n## Condition number", round(np.linalg.cond(m, p=None), 2))
        condition_number = round(np.linalg.cond(m, p=None), 2)
        print("\n## Variance inflation factors:")
        vifs = np.array([round(vif(m, i), 3) for i in range(m.shape[1])])
        df = pd.DataFrame({'Feature_name': name, 'Mean':mean, 'Std':std, 'Condition_number':np.ones(len(mean))*condition_number, 'Square_roor_Variance_inflation_factor':np.sqrt(vifs)}, index=np.ones(len(mean))*i)
        df_final.append(df)
        for i, label in enumerate(dtxmat.columns):
            print("%-15s" % label, vifs[i])
        sns.heatmap(corr)
        plt.savefig(os.path.join(output_folder, file_name)+'pairwise_correlations.png')
        plt.show()
        plt.close()
        i+=1
    df_final = pd.concat(df_final)
    df_final.to_csv(os.path.join(output_folder, 'stats.csv'))