import argparse
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from qrnn import trainQuantile, predict
from mylib.transformer import transform, inverse_transform
from mylib.tools import *



def main(options):
    variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    preshower = ['probeesEnergyOverSCRawEnergy']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    weight = ['ml_weight']

#    data_key = options.data_key
#    EBEE = options.EBEE 
    data_key = 'data'
    EBEE = 'EE' 
     
    spl = options.split
    if spl in [1, 2]: 
        inputtrain = 'tmp_dfs/weightedsys/df_{}_{}_train_split{}.h5'.format(data_key, EBEE, spl)
        nEvt = 820000
    else: 
        inputtrain = 'weighted_dfs/df_{}_{}_train.h5'.format(data_key, EBEE)
        nEvt = 850000
        print(f"Wrong argument '-s' ('--split'), argument must have value 1 or 2. Now using defalt dataframe {inputtrain}")
#    inputtrain = 'weighted_dfs/df_{}_{}_train.h5'.format(data_key, EBEE)
#    inputtest = 'weighted_dfs/df_{}_{}_test.h5'.format(data_key, EBEE)
   
    #load dataframe
#    nEvt = 850000
    query_preshower = 'probeScEta<-1.653 or probeScEta>1.653'
    df_train = ((pd.read_hdf(inputtrain).loc[:,kinrho+variables+preshower+weight]).query(query_preshower)).sample(nEvt, random_state=100).reset_index(drop=True)
    
    #transform features and targets
    transformer_file = 'data_{}'.format(EBEE)
    df_train.loc[:,kinrho+variables] = transform(df_train.loc[:,kinrho+variables], transformer_file, kinrho+variables)
    print(df_train)

    batch_size = pow(2, 13)
    num_hidden_layers = 5
    num_connected_layers = 2
    num_units = [20, 10, 20, 15, 10]
    act = ['tanh' for _ in range(num_hidden_layers)]
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

    target = preshower[0] 
    features = kinrho + variables 
    print('>>>>>>>>> train for variable {} with features {}'.format(target, features))

    X = df_train.loc[:,features]
    Y = df_train.loc[:,target]
    sample_weight = df_train.loc[:,'ml_weight']

    qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    qweights = compute_qweights(Y, qs, sample_weight)
    print('quantile loss weights: {}'.format(qweights))

    model_file = '{}/{}_{}_{}'.format(modeldir, data_key, EBEE, target)
    history, eval_results = trainQuantile(
        X, Y, 
        qs, qweights, 
        num_hidden_layers, num_units, act, 
        num_connected_layers = num_connected_layers,
        sample_weight = sample_weight,
        l2lam = 1.e-3, 
        opt = 'Adadelta', lr = 0.5, 
        batch_size = batch_size, 
        epochs = 1000, 
        save_file = model_file, 
        )

    train_end = time.time()
    print('evaluation results: ', eval_results)
    print('time spent in training: {} s'.format(train_end-train_start))
    
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

   
   



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
#    requiredArgs.add_argument('-i','--ith_var', action='store', type=int, required=True)
#    requiredArgs.add_argument('-d','--data_key', action='store', type=str, required=True)
#    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-s','--split', action='store', type=int)
    options = parser.parse_args()
    main(options)
