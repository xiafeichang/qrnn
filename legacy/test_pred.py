import os
import argparse
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn import preprocessing 
import pickle as pkl
import gzip

from check_results import clf_reweight
from qrnn import trainQuantile, predict, scale
from clf_Iso import trainClfp0t, trainClf3Cat
from train_SS import compute_qweights
from mylib.transformer import transform, inverse_transform
from mylib.Corrector import Corrector, applyCorrection
from mylib.Shifter import Shifter, applyShift
from mylib.Shifter2D import Shifter2D, apply2DShift
from mylib.tools import *


def main(options):
    if options.var_type == 'Ph':
        variables = ['probePhoIso']
    elif options.var_type == 'Ch':
        variables = ['probeChIso03','probeChIso03worst']
    else: 
        raise ValueError('var_type must be "Ph" (for photon) or "Ch" (for charged)')
    kinrho = ['probePt','probeScEta','probePhi','rho'] 

#    data_key = options.data_key
#    data_key = 'mc'
    EBEE = options.EBEE 
     
    inputdata = 'weighted_dfs/df_{}_{}_Iso_train.h5'.format('data', EBEE)
    inputmc = 'weighted_dfs/df_{}_{}_Iso_test.h5'.format('mc', EBEE)
   
    #load dataframe
#    nEvt = 3500000
    nEvt = options.nEvt
    df_data_raw = (pd.read_hdf(inputdata).loc[:,kinrho+variables]).sample(nEvt, random_state=100).reset_index(drop=True)
    df_mc_raw = (pd.read_hdf(inputmc).loc[:,kinrho+variables]).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_train = ((pd.read_hdf('from_massi/weighted_dfs/df_mc_EB_Iso_test.h5').loc[:,kinrho+variables])[:nEvt]).reset_index(drop=True)
     
#    modeldir = 'backup_models'
    modeldir = 'test/chained_models'
    plotsdir = 'test/plots'


    #transform features and targets
    transformer_file = 'data_{}'.format(EBEE)
    df_data_raw.loc[:,kinrho] = transform(df_data_raw.loc[:,kinrho], transformer_file, kinrho)
    df_mc_raw.loc[:,kinrho] = transform(df_mc_raw.loc[:,kinrho], transformer_file, kinrho)

    df_mc_raw['weight_clf'] = clf_reweight(df_mc_raw, df_data_raw, f'transformer/4d_reweighting_{EBEE}_Iso', n_jobs=10)

    qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    qweights = np.ones_like(qs)


    if len(variables)>1: 
        clf_name_data = '{}/data_{}_clf_{}_{}.pkl'.format(modeldir, EBEE, variables[0], variables[1])
        clf_name_mc = '{}/mc_{}_clf_{}_{}.pkl'.format(modeldir, EBEE, variables[0], variables[1])
    else: 
        clf_name_data = '{}/data_{}_clf_{}.pkl'.format(modeldir, EBEE, variables[0])
        clf_name_mc = '{}/mc_{}_clf_{}.pkl'.format(modeldir, EBEE, variables[0])

    if len(variables)>1:
        tReg_models = {}
        for target in variables: 
            model_file_tReg = '{}/{}_{}_tReg_{}'.format(modeldir, 'mc', EBEE, target)
            tReg_models[target] = model_file_tReg

    for target in variables: 
        if not all([(f'{var}_shift' in df_mc_raw.columns) for var in variables]):
            if len(variables)>1: 

                print(f'shifting mc with classifier and tail regressors: {clf_name_mc}, {clf_name_data}, {tReg_models}')
                Y_test_shifted = parallelize(apply2DShift, 
                    df_mc_raw.loc[:,kinrho], df_mc_raw.loc[:,variables],
                    load_clf(clf_name_mc), load_clf(clf_name_data), 
                    tReg_models[variables[0]], tReg_models[variables[1]],
                    qs,qweights,
                    final_reg = False,
                    ) 
                df_mc_raw['{}_shift'.format(variables[0])] = Y_test_shifted[:,0]
                df_mc_raw['{}_shift'.format(variables[1])] = Y_test_shifted[:,1]

            else: 
                print(f'shifting mc with classifiers and tail regressor: {clf_name_mc}, {clf_name_data}, {model_file_mc}')
  
        features_mc = kinrho + ['{}_corr'.format(x) for x in variables[:variables.index(target)]]

        model_file_mc = '{}/{}_{}_{}'.format(modeldir, 'mc', EBEE, target)
        model_file_data = '{}/{}_{}_{}'.format(modeldir, 'data', EBEE, target)
   
        df_mc_raw['{}_corr'.format(target)] = parallelize(applyCorrection, 
            df_mc_raw.loc[:,features_mc], df_mc_raw.loc[:,'{}_shift'.format(target)], 
            model_file_mc, model_file_data, 
            diz=True, 
            )

        if True: #target == 'probeChIso03worst': 
            df_mc_raw_tail = df_mc_raw.query(f'{target}_shift!=0').reset_index(drop=True)
        else: 
            df_mc_raw_tail = df_mc_raw.query(f'{target}!=0').reset_index(drop=True)


        features_data = kinrho + [x for x in variables[:variables.index(target)]]

        df_data_raw_tail = df_data_raw.query(f'{target}!=0').reset_index(drop=True)

        pTs = transform(np.array([25., 30., 35., 40., 45., 50., 60., 150.]), transformer_file, 'probePt')
        for i in range(len(pTs)-1): 
            df_data = df_data_raw_tail.query('probePt>' + str(pTs[i]) + ' and probePt<' + str(pTs[i+1]))
            X_data = df_data.loc[:,features_data]
         
            Y_data = predict(X_data, qs, qweights, model_file_data)
            q_data = np.mean(Y_data, axis=0)
            q_data_err = np.std(Y_data, axis=0)

            df_mc = df_mc_raw_tail.query('probePt>' + str(pTs[i]) + ' and probePt<' + str(pTs[i+1]))
            X_mc = df_mc.loc[:,features_mc]

            Y_mc = predict(X_mc, qs, qweights, model_file_mc)
            q_mc = np.average(Y_mc, axis=0, weights=df_mc['weight_clf'])
            q_mc_err = np.std(Y_mc, axis=0)

            fig = plt.figure(tight_layout=True)
            plt.hist(df_data[target], bins=500, density=True, cumulative=True, histtype='step', color='blue', label='data true')
            plt.hist(df_mc[f'{target}_shift'], bins=500, density=True, cumulative=True, histtype='step', color='red', label='mc true')
            plt.errorbar(q_data, qs, xerr=q_data_err, fmt='.', color='cyan', markersize=7, elinewidth=2, capsize=3, markeredgewidth=2, label='data pred')
            plt.errorbar(q_mc, qs, xerr=q_mc_err, fmt='.', color='violet', markersize=7, elinewidth=2, capsize=3, markeredgewidth=2, label='mc pred')
            plt.xlim(0., 10.)
            plt.xlabel(target)
            plt.legend()
            fig.savefig('{}/test_pred/{}_{}_{}.png'.format(plotsdir, EBEE, target, str(i)))
            fig.savefig('{}/test_pred/{}_{}_{}.pdf'.format(plotsdir, EBEE, target, str(i)))
            plt.close(fig)






if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
#    requiredArgs.add_argument('-d','--data_key', action='store', type=str, required=True)
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-n','--nEvt', action='store', type=int, required=True)
    requiredArgs.add_argument('-v','--var_type', action='store', type=str, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-r','--retrain', action='store', type=str)
    options = parser.parse_args()
    main(options)
