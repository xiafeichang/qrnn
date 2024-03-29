import argparse
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from nn import trainNN, predict
from mylib.transformer import fit_standard_scaler, transform, inverse_transform


def main(options):
    variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    weight = ['ml_weight']

    EBEE = options.EBEE 
     
    spl = options.split
    if spl in [1, 2]: 
        inputmc = 'dfs_sys/split{}/df_mc_{}_train_split{}_corr.h5'.format(spl, EBEE, spl)
    else: 
        inputmc = 'dfs_corr/df_mc_{}_train_corr.h5'.format(EBEE)
        print(f"Wrong argument '-s' ('--split'), argument must have value 1 or 2. Now using defalt dataframe {inputmc}")
#    inputdata = 'weighted_dfs/df_data_{}_train.h5'.format(EBEE)
#    inputmc = 'dfs_corr/df_mc_{}_train_corr.h5'.format(EBEE)
   
    #load dataframe
    nEvt = options.nEvt
#    nEvt = 1000000
    df_train = (pd.read_hdf(inputmc)).sample(nEvt, random_state=100).reset_index(drop=True)

    vars_corr_diff = [f'{var}_corr_diff' for var in variables]
    df_target = pd.concat([df_train[f'{var}_corr']-df_train[var] for var in variables], axis=1).rename(columns={i:vars_corr_diff[i] for i in range(len(vars_corr_diff))})
    
    #transform features and targets
    transformer_file = 'data_{}'.format(EBEE)
    df_train.loc[:,kinrho+variables] = transform(df_train.loc[:,kinrho+variables], transformer_file, kinrho+variables)

    trans_file_cd = f'mc_{EBEE}'
#    try: 
#        df_target = transform(df_target, trans_file_cd, vars_corr_diff)
#    except FileNotFoundError: 
#        fit_standard_scaler(df_target, vars_corr_diff, trans_file_cd)
#        df_target = transform(df_target, trans_file_cd, vars_corr_diff)
    df_target = transform(df_target, trans_file_cd, vars_corr_diff)
 


    batch_size = pow(2, 13)
#    num_hidden_layers = 5
#    num_units = [160, 120, 100, 80, 50]
    num_hidden_layers = 10
    num_units = [30-1*i for i in range(num_hidden_layers)]
    act = ['tanh' for _ in range(num_hidden_layers)]
#    act = ['tanh','exponential', 'softplus', 'elu', 'tanh']
    dropout = [0.1, 0.1, 0.1, 0.1, 0.1]
    gauss_std = [0.1, 0.1, 0.1, 0.1, 0.1]

    train_start = time.time()

    if spl in [1, 2]: 
        modeldir = f'models/split{spl}'
        plotsdir = f'plots/split{spl}'
    else:
        modeldir = 'chained_models'
        plotsdir = 'plots'

#    modeldir = 'chained_models'
#    plotsdir = 'plots'

    sample_weight = df_train.loc[:,'ml_weight']
    features = kinrho + variables

    X = df_train.loc[:,features]
    Y = df_target

    model_file = '{}/mc_{}_SS_final'.format(modeldir, EBEE)
    history, eval_results = trainNN(
        X, Y, 
        num_hidden_layers, num_units, act, 
        sample_weight = sample_weight,
        l2lam = 1.e-3, 
        opt = 'Adadelta', lr = 0.5, 
        batch_size = batch_size, 
        epochs = 1000, 
        checkpoint_dir = f'./ckpt/final/{EBEE}', 
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
    history_fig.savefig('{}/training_histories/mc_{}_final.png'.format(plotsdir, EBEE))




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-n','--nEvt', action='store', type=int, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-s','--split', action='store', type=int)
    options = parser.parse_args()
    main(options)
