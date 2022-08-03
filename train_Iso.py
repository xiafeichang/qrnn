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
from clf_Iso import trainClfp0t, trainClf3Cat
from train_SS import compute_qweights
from mylib.transformer import transform, inverse_transform


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

#    data_key = options.data_key
    data_key = 'data'
    EBEE = options.EBEE 
     
    spl = options.split
    if spl in [1, 2]: 
        inputtrain = 'tmp_dfs/weightedsys/df_{}_{}_Iso_train_split{}.h5'.format(data_key, EBEE, spl)
    else: 
        inputtrain = 'tmp_dfs/weighted0.9/df_{}_{}_Iso_train.h5'.format(data_key, EBEE)
        print(f'Wrong argument "-s" ("--split"), argument must have value 1 or 2. Now using defalt dataframe {inputtrain}')
#    inputtrain = 'tmp_dfs/weighted0.9/df_{}_{}_Iso_train.h5'.format(data_key, EBEE)
   
    #load dataframe
#    nEvt = 3500000
    nEvt = options.nEvt
    df_train = (pd.read_hdf(inputtrain).loc[:,kinrho+variables+weight]).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_train = ((pd.read_hdf('from_massi/weighted_dfs/df_data_EB_Iso_test.h5').loc[:,kinrho+variables+weight])[:nEvt]).reset_index(drop=True)
    print(df_train)
    
    if spl in [1, 2]: 
        modeldir = f'models/split{spl}'
        plotsdir = f'plots/split{spl}'
    else:
        modeldir = 'chained_models'
        plotsdir = 'plots'



    # train classifier for peak or tail
    clf_start = time.time()
    if len(variables)>1: 
        eval_metric='mlogloss'
        clf_results = trainClf3Cat(
            df_train, 
#            kinrho, variables, 
            kinrho, ['probeChIso03','probeChIso03worst'], 
            clf_name = '{}/{}_{}_clf_{}_{}.pkl'.format(modeldir, data_key, EBEE, variables[0], variables[1]),
            tree_method = 'gpu_hist',
#            eval_metric = eval_metric,
#            early_stopping_rounds = 10,
            )
        fig_name = '{}/training_histories/{}_{}_clf_{}_{}.png'.format(plotsdir, data_key, EBEE, variables[0], variables[1])
    else: 
        eval_metric='logloss'
        clf_results = trainClfp0t(
            df_train, 
            kinrho, variables[0], 
            clf_name = '{}/{}_{}_clf_{}.pkl'.format(modeldir, data_key, EBEE, variables[0]),
            tree_method = 'gpu_hist',
#            eval_metric = eval_metric,
#            early_stopping_rounds = 10,
            )
        fig_name = '{}/training_histories/{}_{}_clf_{}.png'.format(plotsdir, data_key, EBEE, variables[0])

    print('time spent in training classifier: {} s'.format(time.time()-clf_start))

#    # plot training history
#    clf_lc_fig = plt.figure(tight_layout=True)
#    plt.plot(clf_results['validation_0'][eval_metric], label='training')
#    plt.plot(clf_results['validation_1'][eval_metric], label='validation')
#    plt.title('Training history')
#    plt.xlabel('epoch')
#    plt.ylabel('log loss')
#    plt.legend()
#    clf_lc_fig.savefig(fig_name)

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
    act = ['tanh' for _ in range(num_hidden_layers)]
    dropout = [0.1, 0.1, 0.1, 0.1, 0.1]

    train_start = time.time()

    for target in variables: 
        features = kinrho + variables[:variables.index(target)] 
        print('>>>>>>>>> train for variable {} with features {}'.format(target, features))
    
        # split dataset into tail part and peak part
        df_train_tail = df_train.query(target+'!=0').reset_index(drop=True)
#        df_train_tail.loc[:,variables] = transform(df_train_tail.loc[:,variables], transformer_file, variables)
#        print(df_train_tail)
    
        # qrnn for tail distribution
        X_tail = df_train_tail.loc[:,features]
        Y_tail = df_train_tail.loc[:,target]
        sample_weight_tail = df_train_tail.loc[:,'ml_weight']
    
        qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
        qweights = compute_qweights(Y_tail, qs, sample_weight_tail)
        print('quantile loss weights: {}'.format(qweights))
    
        model_file = '{}/{}_{}_{}'.format(modeldir, data_key, EBEE, target)
        history, eval_results = trainQuantile(
            X_tail, Y_tail, 
            qs, qweights, 
            num_hidden_layers, num_units, act, 
            num_connected_layers = num_connected_layers,
            sample_weight = sample_weight_tail,
            l2lam = 1.e-3, 
            opt = 'Adadelta', lr = 0.5, 
#            op_act = 'softplus', 
            batch_size = batch_size, 
            epochs = 1000, 
            save_file = model_file, 
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

    train_end = time.time()
    print('time spent in training: {} s'.format(train_end-train_start))
 
   
   



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
#    requiredArgs.add_argument('-d','--data_key', action='store', type=str, required=True)
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-n','--nEvt', action='store', type=int, required=True)
    requiredArgs.add_argument('-v','--var_type', action='store', type=str, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-s','--split', action='store', type=int)
    options = parser.parse_args()
    main(options)
