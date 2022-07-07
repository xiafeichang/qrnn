import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import keras_tuner as kt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint

from mylib.transformer import transform, inverse_transform
from qrnn import *


# GPU memory setup
gpus = tf.config.list_physical_devices('GPU')

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

#if gpus:
#  # Restrict TensorFlow to only allocate 2GB of memory on each GPU
#  try:
#    for gpu in gpus:
#        tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
#    logical_gpus = tf.config.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#  except RuntimeError as e:
#    # Virtual devices must be set before GPUs have been initialized
#    print(e)

class qrnnHyperModel(kt.HyperModel): 

    def __init__(self, qs, qweights, input_dim):
        self.qs = qs
        self.qweights = qweights
        self.input_dim = input_dim

    def build(self, hp):
        
        # cleanup
        try:
            K.clear_session()
            print('Keras backend session cleared!')
        except:
            print('Failed to clear session, continue') 

        qs = self.qs
        qweights = self.qweights
        input_dim = self.input_dim
        
        num_hidden_layers = 6
        opt = 'Adadelta'
#        act = ['tanh','exponential', 'softplus', 'tanh', 'elu']
        act = ['tanh' for _ in range(num_hidden_layers)]
        dp_on = True 
#        dp_on = hp.Boolean('dropout', default=False)
        lr = 0.1
#        lr = hp.Float('learning_rate', 1.e-2, 1.e1, sampling='log', default=10.)

        num_connected_layers = hp.Int('num_connected_layers', 1, 5, default=1)
        l2lam = hp.Float('l2_lambda', 1.e-8, 1.e-2, sampling='log', default=1.e-4)

        num_units = [hp.Int(f'units_{i}', 5, 50, sampling='log') for i in range(num_hidden_layers)]
        dp = [hp.Float(f'dropout_{i}', 0., 0.3, 0.05) for i in range(num_hidden_layers)]
        
        # create a MirroredStrategy
        print('devices: ', tf.config.list_physical_devices('GPU'))
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = get_compiled_model(qs, qweights, input_dim, num_hidden_layers, num_units, act, num_connected_layers, l2lam, opt, lr, dp_on, dp)

        return model 


    def fit(self, hp, model, *args, **kwargs): 
        batch_size = pow(2, 13)
        return model.fit(*args,
                         batch_size = batch_size, 
                         **kwargs,
                        )


def compute_qweights(sr, qs, weights=None):
    quantiles = np.quantile(sr, qs)
    es = np.array(sr)[:,None] - quantiles
    huber_e = Hubber(es, 1.e-4, signed=True)
    loss = np.maximum(qs*huber_e, (qs-1.)*huber_e)
    qweights = 1./np.average(loss, axis=0, weights=weights)
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
    weight = ['ml_weight']

    data_key = options.data_key
    EBEE = options.EBEE 

    print(f'for {data_key}, {EBEE}')
      
    inputtrain = 'weighted_dfs/df_{}_{}_train.h5'.format(data_key, EBEE)
    inputtest = 'weighted_dfs/df_{}_{}_test.h5'.format(data_key, EBEE)
    
    #load dataframe
    nEvt = 2000000
    df_train = (pd.read_hdf(inputtrain).loc[:,kinrho+variables+weight]).sample(nEvt, random_state=100).reset_index(drop=True)
    
    #transform features and targets
    transformer_file = 'data_{}'.format(EBEE)
    df_train.loc[:,kinrho+variables] = transform(df_train.loc[:,kinrho+variables], transformer_file, kinrho+variables)
    print(df_train)

    target = variables[options.ith_var]
    features = kinrho 
    print('>>>>>>>>> run bayesian optimization for variable {} with features {}'.format(target, features))

    X = df_train.loc[:,features]
    Y = df_train.loc[:,target]
    sample_weight = df_train.loc[:,'ml_weight']

    qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    qweights = compute_qweights(Y, qs, sample_weight)
    print('quantile loss weights: {}'.format(qweights))

    checkpoint_dir = 'ckpt/{}_{}_{}'.format(data_key, EBEE, target)
    input_dim = len(X.keys())

    tuner = kt.BayesianOptimization(
        qrnnHyperModel(qs, qweights, input_dim),
        objective = 'val_loss',
        max_trials = 50,
        executions_per_trial = 1,
        seed = 100,
        directory = 'bayesOpt',
        project_name = '{}_{}_{}'.format(data_key, EBEE, target),
        )
    
    search_start = time.time()
    tuner.search(
        X, Y,
        sample_weight = sample_weight, 
        epochs = 300,
        validation_split = 0.1,  
        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=7, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=0.0001, patience=3, verbose=1), 
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

    tf.keras.backend.clear_session()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-i','--ith_var', action='store', type=int, required=True)
    requiredArgs.add_argument('-d','--data_key', action='store', type=str, required=True)
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    options = parser.parse_args()
    main(options)

