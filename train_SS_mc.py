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
import pickle
import gzip

from qrnn import trainQuantile, predict, scale
from mylib.transformer import transform, inverse_transform
from mylib.Corrector import Corrector, applyCorrection
from mylib.tools import *

   

def main(options):
    variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
#    variables = ['probePhiWidth','probeEtaWidth','probeSigmaIeIe','probeS4','probeR9','probeCovarianceIeIp']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    weight = ['ml_weight']

    if options.retrain is not None: 
        retrain = options.retrain.lower() == 'yes' or options.retrain.lower() == 'y'
        print('retrain : {} -> {}'.format(options.retrain, retrain))
    else:
        print('retrain argument not found, set to be False')
        retrain = False

    data_key = 'mc'
    EBEE = options.EBEE 
     
    spl = options.split
    if spl in [1, 2]: 
        inputtrain = 'tmp_dfs/weightedsys/df_{}_{}_train_split{}.h5'.format(data_key, EBEE, spl)
    else: 
        inputtrain = 'weighted_dfs/df_{}_{}_train.h5'.format(data_key, EBEE)
        print(f"Wrong argument '-s' ('--split'), argument must have value 1 or 2. Now using defalt dataframe {inputtrain}")
#    inputtrain = 'weighted_dfs/df_{}_{}_train.h5'.format(data_key, EBEE)
#    inputtest = 'weighted_dfs/df_{}_{}_test.h5'.format(data_key, EBEE)
   
    #load dataframe
    nEvt = options.nEvt
    df_train = (pd.read_hdf(inputtrain).loc[:,kinrho+variables+weight]).sample(nEvt, random_state=100).reset_index(drop=True)
    
    #transform features and targets
    transformer_file = 'data_{}'.format(EBEE)
    df_train.loc[:,kinrho+variables] = transform(df_train.loc[:,kinrho+variables], transformer_file, kinrho+variables)

    # setup neural net
    batch_size = pow(2, 14)
#    num_hidden_layers = 6
#    num_connected_layers = 3
#    num_units = [30, 25, 20, 30, 25, 10]
    num_hidden_layers = 5
    num_connected_layers = 2
    num_units = [30, 15, 20, 15, 10]
    act = ['tanh' for _ in range(num_hidden_layers)]
#    act = ['tanh','exponential', 'softplus', 'elu', 'tanh']
    dropout = [0.1, 0.1, 0.1, 0.1, 0.1]
    gauss_std = [0.2, 0.2, 0.2, 0.2, 0.2]

    #train
    
    train_start = time.time()

    if spl in [1, 2]: 
        modeldir = f'models/split{spl}'
        plotsdir = f'plots/split{spl}'
    else:
        modeldir = 'chained_models'
        plotsdir = 'plots'

    sample_weight = df_train.loc[:,'ml_weight']

#    target = variables[options.ith_var]
#    features = kinrho 
    for target in variables:
        features = kinrho + ['{}_corr'.format(x) for x in variables[:variables.index(target)]]
        print('>>>>>>>>> train for variable {} with features {}'.format(target, features))

        X = df_train.loc[:,features]
        Y = df_train.loc[:,target]


        qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
        qweights = compute_qweights(Y, qs, sample_weight)
        print('quantile loss weights: {}'.format(qweights))

        model_file_mc = '{}/{}_{}_{}'.format(modeldir, 'mc', EBEE, target)
        model_file_data = '{}/{}_{}_{}'.format(modeldir, 'data', EBEE, target)
        if os.path.exists(model_file_mc) and not retrain:  
            print(f'models {model_file_data}, {model_file_mc} already exist, skip training')
        else: 
            print(f'training new mc model for {target}')
            history, eval_results = trainQuantile(
                X, Y, 
                qs, qweights, 
                num_hidden_layers, num_units, act, 
                num_connected_layers = num_connected_layers,
                sample_weight = sample_weight,
                l2lam = 1.e-3, 
                opt = 'Adadelta', lr = 0.5, 
                batch_size = batch_size, 
                epochs = 2, 
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

        print(f'correcting mc with models: {model_file_data}, {model_file_mc}')
        df_train['{}_corr'.format(target)] = parallelize(applyCorrection, X, Y, model_file_mc, model_file_data, diz=False)

    train_end = time.time()
    print('time spent in training: {} s'.format(train_end-train_start))
 




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
#    requiredArgs.add_argument('-i','--ith_var', action='store', type=int, required=True)
#    requiredArgs.add_argument('-d','--data_key', action='store', type=str, required=True)
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-n','--nEvt', action='store', type=int, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-r','--retrain', action='store', type=str)
    optArgs.add_argument('-s','--split', action='store', type=int)
    options = parser.parse_args()
    main(options)
