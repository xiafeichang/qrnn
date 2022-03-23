import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint

from transformer import fit_power_transformer, fit_quantile_transformer, transform, inverse_transform
from qrnn import *


class qrnnHyperModel(kt.HyperModel): 

    def __init__(self, qs, qweights, input_dim):
        self.qs = qs
        self.qweights = qweights
        self.input_dim = input_dim

    def build(self, hp):
        
        qs = self.qs
        qweights = self.qweights
        input_dim = self.input_dim
        
        # for test
#        num_connected_layers = 1
#        num_hidden_layers = 5
#        l2lam = 1.e-5
#        opt = 'RMSprop'
#        lr = hp.Float('learning_rate', 1.e-3, 1., sampling='log')
#        dp_on = False
#        num_units = [10 for _ in range(num_hidden_layers)]
#        act = ['tanh' for _ in range(num_hidden_layers)]
#        dp = [0.1 for _ in range(num_hidden_layers)]

        num_connected_layers = hp.Int('num_connected_layers', 1, 3)
        num_hidden_layers = num_connected_layers + hp.Int('num_disconnected_layers', 2, 5)
        l2lam = hp.Float('l2_lambda', 1.e-7, 1.e-2, sampling='log', default=1.e-4)
        opt = hp.Choice('optimizer', ['RMSprop','Adam','Adadelta','SGD'])
        lr = hp.Float('learning_rate', 1.e-3, 1., sampling='log', default=1.)
        dp_on = hp.Boolean('dropout', default=False)

        num_units = []
        act = []
        dp = []
        for i in range(num_connected_layers):
            num_units.append(hp.Int(f'units_{i}', 500, 2000, 150))
            act.append(hp.Choice(f'activation_{i}', ['relu', 'tanh']))
            dp.append(hp.Float(f'dropout_{i}', 0.05, 0.3, 0.05))
        for i in range(num_connected_layers, num_hidden_layers):
            num_units.append(hp.Int(f'units_{i}', 5, 50, 5))
            act.append(hp.Choice(f'activation_{i}', ['relu', 'tanh']))
            dp.append(hp.Float(f'dropout_{i}', 0.05, 0.3, 0.05))

        return get_compiled_model(qs, qweights, input_dim, num_hidden_layers, num_units, act, num_connected_layers, l2lam, opt, lr, dp_on, dp)

    def fit(self, hp, model, *args, **kwargs): 
        return model.fit(*args,
                         batch_size = pow(2, hp.Int('bacth_size_pow', 5, 17, default=10))
                         **kwargs,
                        )


def compute_qweights(sr, qs):
    quantiles = np.quantile(sr, qs)
    es = np.array(sr)[:,None] - quantiles
    huber_e = Hubber(es, 1.e-4, signed=True)
    loss = np.maximum(qs*huber_e, (qs-1.)*huber_e)
    qweights = 1./np.mean(loss, axis=0)
    return qweights/np.min(qweights)

def Hubber(e, delta=0.1, signed=False):
    is_small_e = np.abs(e) < delta
    small_e = np.square(e) / (2.*delta)
    big_e = np.abs(e) - delta/2.
    if signed:
        return np.sign(e)*np.where(is_small_e, small_e, big_e) 
    else: 
        return np.where(is_small_e, small_e, big_e)


def main(options): 

    # prepare data set
    variables = ['probeS4','probeR9','probeCovarianceIeIp','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 

    data_key = 'mc'
    EBEE = 'EB'
      
    inputtrain = 'df_{}_{}_train.h5'.format(data_key, EBEE)
    inputtest = 'df_{}_{}_test.h5'.format(data_key, EBEE)
    
    #load dataframe
    nEvt = 500000
    df_train = (pd.read_hdf(inputtrain).loc[:,kinrho+variables]).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_test_raw  = (pd.read_hdf(inputtest).loc[:,kinrho+variables]).sample(nEvt, random_state=100).reset_index(drop=True)
    
    #scale or transform data
    scale_file = 'scale_para/data_{}.h5'.format(EBEE)
    scale_par = pd.read_hdf(scale_file)

    transformer_file = 'data_{}'.format(EBEE)
    df_train.loc[:,variables] = transform(df_train.loc[:,variables], transformer_file, variables)
#    df_test_raw.loc[:,variables] = transform(df_test_raw.loc[:,variables], transformer_file, variables)

    df_train.loc[:,kinrho] = scale(df_train.loc[:,kinrho], scale_file=scale_file)
#    df_test_raw.loc[:,kinrho] = scale(df_test_raw.loc[:,kinrho], scale_file=scale_file)

    print(df_train)
#    print(df_test_raw)

    target = variables[options.ith_var]
    features = kinrho 
    print('>>>>>>>>> run bayesian optimization for variable {} with features {}'.format(target, features))

    X = df_train.loc[:,features]
    Y = df_train.loc[:,target]
#    X_test_raw = df_test_raw.loc[:,features]
#    Y_test_raw = df_test_raw.loc[:,target]

    
    qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    qweights = compute_qweights(Y, qs)
    print('quantile loss weights: {}'.format(qweights))

    checkpoint_dir = 'ckpt/{}_{}_{}'.format(data_key, EBEE, target)
    input_dim = len(X.keys())

    tuner = kt.BayesianOptimization(
        qrnnHyperModel(qs, qweights, input_dim),
        objective = 'val_loss',
        max_trials = 30,
        seed = 100,
        directory = 'bayesOpt',
        project_name = '{}_{}_{}'.format(data_key, EBEE, target)
        )
    
    search_start = time.time()
    tuner.search(
        X, Y,
        epochs = 1000,
        validation_split = 0.1,  
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, verbose=1), 
            ModelCheckpoint(filepath=checkpoint_dir+'/ckpt-{epoch}', save_freq="epoch"),
            TerminateOnNaN()
            ], 
        shuffle = True,
        )
    search_end = time.time()
    print('time spent in searching: {} s'.format(search_end-search_start))
    
    models = tuner.get_best_models(num_models=2)
    best_model = models[0]
    best_model.summary()
    best_model.save('best_models/{}_{}_{}'.format(data_key, EBEE, target))

    tuner.results_summary()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-i','--ith_var', action='store', type=int, required=True)
    options = parser.parse_args()
    main(options)

