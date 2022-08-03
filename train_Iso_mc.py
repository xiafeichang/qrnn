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
#        variables = ['probeChIso03','probeChIso03worst']
        variables = ['probeChIso03worst','probeChIso03']
    else: 
        raise ValueError('var_type must be "Ph" (for photon) or "Ch" (for charged)')
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    weight = ['ml_weight']

    if options.retrain is not None: 
        retrain = options.retrain.lower() == 'yes' or options.retrain.lower() == 'y'
        print('retrain : {} -> {}'.format(options.retrain, retrain))
    else:
        print('retrain argument not found, set to be False')
        retrain = False

#    data_key = options.data_key
    data_key = 'mc'
    EBEE = options.EBEE 
     
    spl = options.split
    if spl in [1, 2]: 
        inputtrain = 'tmp_dfs/weightedsys/df_{}_{}_Iso_train_split{}.h5'.format(data_key, EBEE, spl)
    else: 
        inputtrain = 'tmp_dfs/weighted0.9/df_{}_{}_Iso_train.h5'.format(data_key, EBEE)
        print(f"Wrong argument '-s' ('--split'), argument must have value 1 or 2. Now using defalt dataframe {inputtrain}")
#    inputtrain = 'tmp_dfs/weighted0.9/df_{}_{}_Iso_train.h5'.format(data_key, EBEE)
   
    #load dataframe
#    nEvt = 3500000
    nEvt = options.nEvt
    df_train = (pd.read_hdf(inputtrain).loc[:,kinrho+variables+weight]).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_train = ((pd.read_hdf('from_massi/weighted_dfs/df_mc_EB_Iso_test.h5').loc[:,kinrho+variables])[:nEvt]).reset_index(drop=True)
     
    if spl in [1, 2]: 
        modeldir = f'models/split{spl}'
        plotsdir = f'plots/split{spl}'
    else:
        modeldir = 'chained_models'
        plotsdir = 'plots'


    # train classifier for peak or tail
    clf_start = time.time()
    if len(variables)>1: 
        clf_name_data = '{}/data_{}_clf_{}_{}.pkl'.format(modeldir, EBEE, variables[0], variables[1])
        clf_name_mc = '{}/mc_{}_clf_{}_{}.pkl'.format(modeldir, EBEE, variables[0], variables[1])
        if os.path.exists(clf_name_mc) and not retrain:
            print(f'{clf_name_mc} already exist, skip training')
        else: 
            print('>>>>>>>>> train classifier for variable(s) {} with features {}'.format(variables, kinrho))
            clf_results = trainClf3Cat(
                df_train, 
#                kinrho, variables, 
                kinrho, ['probeChIso03','probeChIso03worst'], 
                clf_name = clf_name_mc,
                tree_method = 'gpu_hist',
#                eval_metric='mlogloss',
                )
            fig_name = '{}/training_histories/{}_{}_clf_{}_{}.png'.format(plotsdir, data_key, EBEE, variables[0], variables[1])
    else: 
        clf_name_data = '{}/data_{}_clf_{}.pkl'.format(modeldir, EBEE, variables[0])
        clf_name_mc = '{}/mc_{}_clf_{}.pkl'.format(modeldir, EBEE, variables[0])
        if os.path.exists(clf_name_mc) and not retrain:
            print(f'{clf_name_mc} already exist, skip training')
        else: 
            print('>>>>>>>>> train classifier for variable(s) {} with features {}'.format(variables, kinrho))
            clf_results = trainClfp0t(
                df_train, 
                kinrho, variables[0], 
                clf_name = clf_name_mc,
                tree_method = 'gpu_hist',
#                eval_metric='logloss',
                )
            fig_name = '{}/training_histories/{}_{}_clf_{}.png'.format(plotsdir, data_key, EBEE, variables)

    print('time spent in training classifier: {}-{:02d}:{:02d}:{:05.2f}'.format(*sec2HMS(time.time()-clf_start)))

    # plot training history
#    try: 
#        clf_lc_fig = plt.figure(tight_layout=True)
#        plt.plot(clf_results['validation_0']['logloss'], label='training')
#        plt.plot(clf_results['validation_1']['logloss'], label='validation')
#        plt.title('Training history')
#        plt.xlabel('epoch')
#        plt.ylabel('log loss')
#        plt.legend()
#        clf_lc_fig.savefig(fig_name)
#    except: 
#        print('Failed to draw learning curve for classifier training. Check if skipped because they are already exist')


    #transform features and targets
    transformer_file = 'data_{}'.format(EBEE)
    df_train.loc[:,kinrho] = transform(df_train.loc[:,kinrho], transformer_file, kinrho)

    # train qrnn
    batch_size = pow(2, 13)
    num_hidden_layers = 5
    num_connected_layers = 2
    num_units = [30, 15, 20, 15, 10]
#    num_hidden_layers = 6
#    num_connected_layers = 3
#    num_units = [20, 15, 10, 20, 15, 10]
#    num_hidden_layers = 6
#    num_connected_layers = 3
#    num_units = [30, 25, 20, 30, 25, 10]
    act = ['tanh' for _ in range(num_hidden_layers)]
    dropout = [0.1, 0.1, 0.1, 0.1, 0.1]

    qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])


    if len(variables)>1:
        # train tail regressors
        tReg_start = time.time()
        tReg_models = {}
        for target in variables: 
            features = kinrho + [var for var in variables if var != target] 
        
            # split dataset into tail part and peak part
            df_train_tReg = df_train.query(f'{target}!=0').reset_index(drop=True)
        
            # qrnn for tail distribution
            X_tReg = df_train_tReg.loc[:,features]
            Y_tReg = df_train_tReg.loc[:,target]
            sample_weight_tReg = df_train_tReg.loc[:,'ml_weight']
        
            qweights = compute_qweights(Y_tReg, qs, sample_weight_tReg)
            print('quantile loss weights: {}'.format(qweights))
        
            model_file_tReg = '{}/{}_{}_tReg_{}'.format(modeldir, data_key, EBEE, target)
            tReg_models[target] = model_file_tReg
            if os.path.exists(model_file_tReg) and not retrain:
                print(f'{model_file_tReg} already exist, skip training')
            else: 
                print('>>>>>>>>> train tail regressor for variable {} with features {}'.format(target, features))
                history, eval_results = trainQuantile(
                    X_tReg, Y_tReg, 
                    qs, qweights, 
                    num_hidden_layers, num_units, act, 
                    num_connected_layers = num_connected_layers,
                    sample_weight = sample_weight_tReg,
                    l2lam = 1.e-3, 
                    opt = 'Adadelta', lr = 0.5, 
#                    op_act = 'softplus', 
                    batch_size = batch_size, 
                    epochs = 1000, 
                    save_file = model_file_tReg, 
                    )
        
                # plot training history
                history_fig = plt.figure(tight_layout=True)
                plt.plot(history.history['loss'], label='training')
                plt.plot(history.history['val_loss'], label='validation')
                plt.yscale('log')
                plt.title('Training history')
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.legend()
                history_fig.savefig('{}/training_histories/{}_{}_tReg_{}.png'.format(plotsdir, data_key, EBEE, target))
    
        print('time spent in training tail regressors: {}-{:02d}:{:02d}:{:05.2f}'.format(*sec2HMS(time.time()-tReg_start)))
        del df_train_tReg, X_tReg, Y_tReg, sample_weight_tReg # delete them to release memory

    train_start = time.time()

    # train qrnn
    for target in variables: 
        features = kinrho + ['{}_corr'.format(x) for x in variables[:variables.index(target)]]

        model_file_mc = '{}/{}_{}_{}'.format(modeldir, 'mc', EBEE, target)
        model_file_data = '{}/{}_{}_{}'.format(modeldir, 'data', EBEE, target)
   
  
        # qrnn for tail distribution
        if False: #target == 'probeChIso03worst': 
            df_train_tail = df_train.query(f'{target}_shift!=0').reset_index(drop=True)
            Y_tail = df_train_tail.loc[:,f'{target}_shift']
        else: 
            df_train_tail = df_train.query(f'{target}!=0').reset_index(drop=True)
            Y_tail = df_train_tail.loc[:,target]
        X_tail = df_train_tail.loc[:,features]
        sample_weight_tail = df_train_tail.loc[:,'ml_weight']


        qweights = compute_qweights(Y_tail, qs, sample_weight_tail)
        print('quantile loss weights: {}'.format(qweights))
    
        if os.path.exists(model_file_mc) and not retrain:
            print(f'{model_file_mc} already exist, skip training')
        else: 
            print('>>>>>>>>> train for variable {} with features {}'.format(target, features))
            history, eval_results = trainQuantile(
                X_tail, Y_tail, 
                qs, qweights, 
                num_hidden_layers, num_units, act, 
                num_connected_layers = num_connected_layers,
                sample_weight = sample_weight_tail,
                l2lam = 1.e-3, 
                opt = 'Adadelta', lr = 0.5, 
#                op_act = 'softplus', 
                batch_size = batch_size, 
                epochs = 1000, 
                save_file = model_file_mc, 
                )
    
            # plot training history
            history_fig = plt.figure(tight_layout=True)
            plt.plot(history.history['loss'], label='training')
            plt.plot(history.history['val_loss'], label='validation')
            plt.yscale('log')
            plt.title('Training history')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            history_fig.savefig('{}/training_histories/{}_{}_{}.png'.format(plotsdir, data_key, EBEE, target))


        if not all([(f'{var}_shift' in df_train.columns) for var in variables]):
            if len(variables)>1: 
                print(f'shifting mc with classifier and tail regressors: {clf_name_mc}, {clf_name_data}, {tReg_models}')
                # VERY IMPORTANT! Note the order of targets here
                Y_shifted = parallelize(apply2DShift, 
                    df_train.loc[:,kinrho], df_train.loc[:,['probeChIso03','probeChIso03worst']],
                    load_clf(clf_name_mc), load_clf(clf_name_data), 
                    tReg_models['probeChIso03'], tReg_models['probeChIso03worst'],
#                    qs,qweights,
                    final_reg = False,
                    ) 
                df_train['probeChIso03_shift'] = Y_shifted[:,0]
                df_train['probeChIso03worst_shift'] = Y_shifted[:,1]
            else: 
                print(f'shifting mc with classifiers and tail regressor: {clf_name_mc}, {clf_name_data}, {model_file_mc}')
                Y_shifted = parallelize(applyShift, 
                    df_train.loc[:,kinrho], df_train.loc[:,variables[0]],
                    load_clf(clf_name_mc), load_clf(clf_name_data), 
                    model_file_mc,
#                    qs,qweights,
                    final_reg = False,
                    ) 
                df_train['{}_shift'.format(variables[0])] = Y_shifted

        print(f'correcting mc with models: {model_file_data}, {model_file_mc}')
        df_train['{}_corr'.format(target)] = parallelize(applyCorrection, 
            df_train.loc[:,features], df_train.loc[:,'{}_shift'.format(target)], 
            model_file_mc, model_file_data, 
            diz=True, 
            )



    train_end = time.time()
    print('time spent in training: {}-{:02d}:{:02d}:{:05.2f}'.format(*sec2HMS(train_end-train_start)))
 
   
   



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
#    requiredArgs.add_argument('-d','--data_key', action='store', type=str, required=True)
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-n','--nEvt', action='store', type=int, required=True)
    requiredArgs.add_argument('-v','--var_type', action='store', type=str, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-r','--retrain', action='store', type=str)
    optArgs.add_argument('-s','--split', action='store', type=int)
    options = parser.parse_args()
    main(options)
