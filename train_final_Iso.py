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
    if options.var_type == 'Ph':
        variables = ['probePhoIso']
    elif options.var_type == 'Ch':
        variables = ['probeChIso03','probeChIso03worst']
    else: 
        raise ValueError('var_type must be "Ph" (for photon) or "Ch" (for charged)')
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    weight = ['ml_weight']

    EBEE = options.EBEE 
     
    spl = options.split
    if spl in [1, 2]: 
        inputmc = 'dfs_sys/split{}/df_mc_{}_Iso_train_split{}_corr.h5'.format(spl, EBEE, spl)
    else: 
#        inputmc = 'dfs_corr/df_mc_{}_Iso_train_corr.h5'.format(EBEE)
        inputmc = 'dfs_sys/df_mc_{}_Iso_train_corr.h5'.format(EBEE)
        print(f"Wrong argument '-s' ('--split'), argument must have value 1 or 2. Now using defalt dataframe {inputmc}")
#    inputdata = 'weighted_dfs/df_data_{}_Iso_train.h5'.format(EBEE)
#    inputmc = 'dfs_corr/df_mc_{}_Iso_train_corr.h5'.format(EBEE)
   
    #load dataframe
    nEvt = options.nEvt
    df_train = (pd.read_hdf(inputmc)).sample(nEvt, random_state=100).reset_index(drop=True)

    #transform features and targets
    transformer_file = 'data_{}'.format(EBEE)
    df_train.loc[:,kinrho] = transform(df_train.loc[:,kinrho], transformer_file, kinrho)


    batch_size = pow(2, 13)
#    num_hidden_layers = 5
#    num_units = [160, 120, 100, 80, 50]
    num_hidden_layers = 10
    num_units = [30-1*i for i in range(num_hidden_layers)]
    act = ['tanh' for _ in range(num_hidden_layers)]
#    act = ['tanh','exponential', 'softplus', 'elu', 'tanh']
    dropout = [0.1, 0.1, 0.1, 0.1, 0.1]
    gauss_std = [0.1, 0.1, 0.1, 0.1, 0.1]

    if spl in [1, 2]: 
        modeldir = f'models/split{spl}'
        plotsdir = f'plots/split{spl}'
    else:
        modeldir = 'chained_models'
        plotsdir = 'plots'

#    modeldir = 'chained_models'
#    plotsdir = 'plots'


    for target in variables: 
        features = kinrho + variables
        print(f'train final regressor for {target} with features: {features}') 
        df_train_tail = df_train.query(f'{target}_corr!=0 and {target}!=0')

        X = df_train_tail.loc[:,features]
    
        target_name = f'{target}_corr_diff'
        Y_raw = df_train_tail[f'{target}_corr']-df_train_tail[target]
        print(Y_raw)
        print(np.mean(Y_raw), np.std(Y_raw))
     
        trans_file_cd = f'mc_{EBEE}'
        try: 
            Y = transform(Y_raw, trans_file_cd, target_name)
        except FileNotFoundError: 
            fit_standard_scaler(Y_raw, target_name, trans_file_cd)
            Y = transform(Y_raw, trans_file_cd, target_name)
#        Y = transform(Y_raw, trans_file_cd, target_name)
        print(Y)
        print(Y.shape, len(Y.shape))
        print(np.mean(Y), np.std(Y))

        print(inverse_transform(Y, trans_file_cd, target_name))
     
        sample_weight = df_train_tail.loc[:,'ml_weight']
    
        model_file = '{}/mc_{}_{}_final'.format(modeldir, EBEE, target)
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
        history_fig.savefig('{}/training_histories/mc_{}_{}_final.png'.format(plotsdir, EBEE, target))




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-n','--nEvt', action='store', type=int, required=True)
    requiredArgs.add_argument('-v','--var_type', action='store', type=str, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-s','--split', action='store', type=int)
    options = parser.parse_args()
    main(options)
