import os
import json
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Input, Dense, GaussianNoise, Dropout, concatenate
from keras.models import Model, load_model
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp
from mylib.transformer import transform, inverse_transform

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




def trainPBNN(X, Y, num_hidden_layers=3, num_units=None, act=None, sample_weight=None, opt='SGD', lr=0.1, dp=None, use_proba_output=False, num_normal_layers=0, batch_size=64, epochs=10, checkpoint_dir='./ckpt', save_file=None, evaluate_data=None, model_plot=None):

    # cleanup
    try:
        K.clear_session()
        print('Keras backend session cleared!')
    except:
        print('Failed to clear session, continue') 

    input_dim = len(X.keys()) 
    output_dim = len(Y.keys())
    train_size = 0.9*len(Y)
    
    if num_units is None: 
        num_units = [10 for i in range(num_hidden_layers)]
    
    if act is None:
        act = ['linear' for i in range(num_hidden_layers)]

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # create a MirroredStrategy
    print('devices: ', tf.config.list_physical_devices('GPU'))
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
#        model = load_or_restore_model(checkpoint_dir, qs, input_dim, num_hidden_layers, num_units, act, qweights, dp, gauss_std)
        print('Creating a new model')
        model = get_compiled_model(input_dim, output_dim, num_hidden_layers, num_units, act, train_size, opt, lr, dp, use_proba_output, num_normal_layers)

    model.summary()
    history = model.fit(
        X, Y, 
        sample_weight = sample_weight, 
        epochs = epochs, 
        batch_size = batch_size, 
        validation_split = 0.1,
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, verbose=1), 
#            ModelCheckpoint(filepath=checkpoint_dir + "/ckpt", save_freq="epoch"),
            TerminateOnNaN()
            ], 
        shuffle = True,
        )


    if save_file is not None:
        model_arch = {'input_dim': input_dim, 'output_dim': output_dim, 'num_hidden_layers': num_hidden_layers, 
                      'num_units': num_units, 'act': act, 'train_size': train_size, 'opt': opt, 'lr': lr, 
                      'dp': dp, 'use_proba_output': use_proba_output, 'num_normal_layers': num_normal_layers}
        with open(f'{save_file}.json', 'w') as f: 
            json.dump(model_arch, f)
        model.save_weights(save_file) 
#        model.save(save_file, save_traces=False) # cannot save traces when there are tfp layers

    eval_results = None
    if evaluate_data is not None:
        eval_results = model.evaluate(evaluate_data[0], evaluate_data[1], batch_size=batch_size)

    if model_plot is not None:
        keras.utils.plot_model(model, to_file=model_plot, show_shapes=True)

    return history, eval_results


def predict(X, model_from=None, nTrials=1, transformer=None, target_name=None):

#    try: 
#        model = load_model(model_from)
#    except: 
#        model = load_model(model_from, custom_objects={'nll':nll})

    with open(f'{model_from}.json', 'r') as f: 
        model_arch = json.load(f)

    with tf.distribute.MirroredStrategy().scope():
        model = get_compiled_model(**model_arch) 
#        model = get_compiled_model(model_arch['input_dim'], model_arch['output_dim'], model_arch['num_hidden_layers'], 
#            model_arch['num_units'], model_arch['act'], model_arch['train_size'], model_arch['opt'], model_arch['lr'], 
#            model_arch['dp'], model_arch['use_proba_output'])

    model.load_weights(model_from)

    predYs = []
    for i in range(nTrials):
        predYs.append(model.predict(X))
        print(predYs[-1])

    if transformer is not None: 
        print('target is transformed, now mapping it back!')
        if isinstance(target_name, list):
            for i in range(nTrials):
                predYs[i] = pd.DataFrame(predYs[i], columns=target_name)
        for i in range(nTrials):
            predYs[i] = inverse_transform(predYs[i], transformer, target_name) 

    return predYs if nTrials>1 else predYs[0]
        
def predictMean(X, model_from=None, transformer=None, target_name=None):

#    model = load_model(model_from, custom_objects={'nll':nll})

    with open(f'{model_from}.json', 'r') as f: 
        model_arch = json.load(f)

    with tf.distribute.MirroredStrategy().scope():
#        model = get_compiled_model(model_arch['input_dim'], model_arch['output_dim'], model_arch['num_hidden_layers'], 
#            model_arch['num_units'], model_arch['act'], model_arch['train_size'], model_arch['opt'], model_arch['lr'], 
#            model_arch['dp'], model_arch['use_proba_output'])
        model = get_compiled_model(**model_arch) 

    model.load_weights(model_from)
    predY = model(X).mean()

    print(predY)
    if transformer is not None: 
        print('target is transformed, now mapping it back!')
        if isinstance(target_name, list):
            predY = pd.DataFrame(predY, columns=target_name)
        predY = inverse_transform(predY, transformer, target_name) 

    return predY
        
def predictStd(X, model_from=None, transformer=None, target_name=None):

#    model = load_model(model_from, custom_objects={'nll':nll})

    with open(f'{model_from}.json', 'r') as f: 
        model_arch = json.load(f)

    with tf.distribute.MirroredStrategy().scope():
        model = get_compiled_model(**model_arch) 
#        model = get_compiled_model(model_arch['input_dim'], model_arch['output_dim'], model_arch['num_hidden_layers'], 
#            model_arch['num_units'], model_arch['act'], model_arch['train_size'], model_arch['opt'], model_arch['lr'], 
#            model_arch['dp'], model_arch['use_proba_output'])

    model.load_weights(model_from)
    predY = model(X).stddev()

    print(predY)
    if transformer is not None: 
        print('target is transformed, now mapping it back!')
        if isinstance(target_name, list):
            predY = pd.DataFrame(predY, columns=target_name)
        predY = inverse_transform(predY, transformer, target_name) 

    return predY
        

def get_compiled_model(input_dim, output_dim, num_hidden_layers, num_units, act, train_size=1, opt='SGD', lr=0.1, dp=None, use_proba_output=False, num_normal_layers=0):

    inpt = Input((input_dim,), name='inpt')

    x = inpt

    for i in range(num_hidden_layers-num_normal_layers): 
        x = tfp.layers.DenseVariational(
            units = num_units[i], 
            make_prior_fn = prior, 
            make_posterior_fn = posterior, 
            kl_weight = 1., #1./train_size
            activation = act[i], 
            )(x)
        if dp is not None:
            x = Dropout(dp[i])(x)

    for i in range(num_hidden_layers-num_normal_layers, num_hidden_layers): 
        x = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1.e-3), 
                  activation=act[i])(x)
        if dp is not None:
            x = Dropout(dp[i])(x)

   
    if use_proba_output: 
        x = Dense(2*output_dim, kernel_initializer='he_normal', bias_initializer='he_normal')(x)
        output = tfp.layers.IndependentNormal(output_dim)(x)
    else: 
        output = Dense(output_dim, activation=None, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(x)

    model = Model(inpt, output)

    # choose optimizer
    if opt.lower() == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif opt.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt.lower() == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr)
    elif opt.lower() == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    elif opt.lower() == 'adamax':
        optimizer = tf.keras.optimizers.Adamax(learning_rate=lr)
    elif opt.lower() == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
    elif opt.lower() == 'ftrl':
        optimizer = tf.keras.optimizers.Ftrl(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    print('compile model with optimizer: ', optimizer)

    if use_proba_output: 
        model.compile(loss=nll, optimizer=optimizer)
    else: 
        model.compile(loss='mse', optimizer=optimizer)
#    model.summary()

    return model


def load_or_restore_model(checkpoint_dir, *args, **kwargs):

    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        return load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model(*args, **kwargs)


def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.ones(n), scale_diag=tf.ones(n)
#                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def nll(y_true, dist_pred): 
    return -dist_pred.log_prob(y_true)


