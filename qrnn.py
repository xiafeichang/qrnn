import os
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Input, Dense, GaussianNoise, Dropout, concatenate
from keras.models import Model, load_model
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
from tensorflow import keras
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)


def trainQuantile(X, Y, qs, qweights=None, num_hidden_layers=3, num_units=None, act=None, num_connected_layers=1, l2lam=1.e-3, opt='SGD', lr=0.1, dp_on=False, dp=None, gauss_std=None, batch_size=64, epochs=10, checkpoint_dir='./ckpt', save_file=None, evaluate_data=None, model_plot=None):

    # cleanup
    try:
        K.clear_session()
        print('Keras backend session cleared!')
    except:
        print('Failed to clear session, continue') 

    input_dim = len(X.keys())
    
    if num_units is None: 
        num_units = [10 for i in range(num_hidden_layers)]
    
    if act is None:
        act = ['linear' for i in range(num_hidden_layers)]

    if qweights is None: 
        qweights = np.ones_like(qs)

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
        model = get_compiled_model(qs, qweights, input_dim, num_hidden_layers, num_units, act, num_connected_layers, l2lam, opt, lr, dp_on, dp, gauss_std)

    history = model.fit(
        X, Y, 
        sample_weight = sample_weight, 
        epochs = epochs, 
        batch_size = batch_size, 
        validation_split = 0.1,
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, verbose=1), 
#            ModelCheckpoint(filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"),
            TerminateOnNaN()
            ], 
        shuffle = True,
        )


    if save_file is not None:
        model.save(save_file)

    eval_results = None
    if evaluate_data is not None:
        eval_results = model.evaluate(evaluate_data[0], evaluate_data[1], batch_size=batch_size)

    if model_plot is not None:
        keras.utils.plot_model(model, to_file=model_plot, show_shapes=True)

    return history, eval_results


def predict(X, qs, qweights, model_from=None, scale_par=None):

    def custom_loss(y_true, y_pred): 
        return qloss(y_true, y_pred, qs, qweights)
    model = load_model(model_from, custom_objects={'custom_loss':custom_loss})

    with tf.distribute.MirroredStrategy().scope():
        predY = model.predict(X)

    if scale_par is not None: 
        logger.info('target is scaled, now mapping it back!')
        predY = predY*scale_par['sigma'] + scale_par['mu']

    return predY
        


def scale(df, scale_file):

    df = pd.DataFrame(df)

    par = pd.read_hdf(scale_file).loc[:,df.keys()] 

    df_scaled = (df - par.loc['mu',:])/par.loc['sigma',:]
    return df_scaled


def get_compiled_model(qs, qweights, input_dim, num_hidden_layers, num_units, act, num_connected_layers=1, l2lam=1.e-3, opt='SGD', lr=0.1, dp_on=False, dp=None, gauss_std=None):

    nq = len(qs)
    
    inpt = Input((input_dim,), name='inpt')

    x = inpt
    
    if dp_on:
        for i in range(num_connected_layers): # fully connected layers
            x = Dense(nq*num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2lam), 
                      activation=act[i])(x)
            x = Dropout(dp[i])(x)
    
        if num_hidden_layers > num_connected_layers: # in following hidden layers, fully connected within one quantile, isolated between quantiles
            xs = []
            for j in range(nq): 
                xs.append(Dense(num_units[num_connected_layers], 
                                use_bias=True, 
                                kernel_initializer='he_normal', 
                                bias_initializer='he_normal',
                                kernel_regularizer=regularizers.l2(l2lam), 
                                activation=act[num_connected_layers])(x))
                xs[j] = Dropout(dp[num_connected_layers])(xs[j])

            if num_hidden_layers > num_connected_layers + 1:
                for i in range(num_connected_layers+1,num_hidden_layers):
                    for j in range(nq):
                        xs[j] = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal',
                                      kernel_regularizer=regularizers.l2(l2lam), 
                                      activation=act[i])(xs[j])
                        xs[j] = Dropout(dp[i])(xs[j])
#                        xs[j] = GaussianNoise(gauss_std[i])(xs[j])  
    else:
        for i in range(num_connected_layers): 
            x = Dense(nq*num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2lam), 
                      activation=act[i])(x)
    
        if num_hidden_layers > num_connected_layers: 
            xs = []
            for j in range(nq): 
                xs.append(Dense(num_units[num_connected_layers], 
                                use_bias=True, 
                                kernel_initializer='he_normal', 
                                bias_initializer='he_normal',
                                kernel_regularizer=regularizers.l2(l2lam), 
                                activation=act[num_connected_layers])(x))

            if num_hidden_layers > num_connected_layers + 1:
                for i in range(num_connected_layers+1,num_hidden_layers):
                    for j in range(nq):
                        xs[j] = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal',
                                      kernel_regularizer=regularizers.l2(l2lam), 
                                      activation=act[i])(xs[j])

    
    # output layer
    try: 
        for j in range(len(qs)): 
            xs[j] = Dense(1, activation=None, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(xs[j])
        output = concatenate(xs)
    except NameError:
        output = Dense(len(qs), activation=None, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(x)

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

    def custom_loss(y_true, y_pred): 
        return qloss(y_true, y_pred, qs, qweights)
    model.compile(loss=custom_loss, optimizer=optimizer)
#    model.summary()

    return model


def load_or_restore_model(checkpoint_dir, qs, qweights, input_dim, num_hidden_layers, num_units, act, num_connected_layers=1, l2lam=1.e-3, opt='SGD', lr=0.1, dp_on=False, dp=None, gauss_std=None):

    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        def custom_loss(y_true, y_pred): 
            return qloss(y_true, y_pred, qs, qweights)
        return load_model(latest_checkpoint, custom_objects={'custom_loss':custom_loss})
    print("Creating a new model")
    return get_compiled_model(qs, qweights, input_dim, num_hidden_layers, num_units, act, num_connected_layers, l2lam, opt, lr, dp_on, dp, gauss_std)


def qloss(y_true, y_pred, qs, qweights):
    q = np.array(qs)
    qweight = np.array(qweights)
    e = y_true - y_pred
    huber_e = Hubber(e, delta=1.e-4, signed=True)
    losses = K.maximum(q*huber_e, (q-1.)*huber_e)*qweight
    return K.mean(losses)

def Hubber(e, delta=0.1, signed=False):
    is_small_e = K.abs(e) < delta
    small_e = K.square(e) / (2.*delta)
    big_e = K.abs(e) - delta/2.
    if signed:
        return K.sign(e)*tf.where(is_small_e, small_e, big_e)
    else:
        return tf.where(is_small_e, small_e, big_e)





