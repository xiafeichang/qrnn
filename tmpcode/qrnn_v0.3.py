import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Input, Dense, GaussianNoise, Dropout
from keras.models import Model, load_model
from keras import regularizers
from tensorflow import keras
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)


def trainQuantile(X, Y, q, num_hidden_layers=1, num_units=None, act=None, batch_size=64, scale_features=None, scale_target=None, save_file=None):

    if scale_features is not None:
        logger.info('Scaling features with {}! '.format(scale_file))
        X = scale(X, scale_features)
    if scale_target is not None:
        Y = scale(Y, scale_target)

    input_dim = len(X.keys())
    
    if num_units is None: 
        num_units = [10 for i in range(num_hidden_layers)]
    
    if act is None:
        act = ['linear' for i in range(num_hidden_layers)]


    inpt = Input((input_dim,), name='inpt')
    
    x = inpt
    
    for i in range(num_hidden_layers):
        x = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal',
                kernel_regularizer=regularizers.l2(0.001), activation=act[i])(x)
         x = Dropout(dp[i])(x)
         x = GaussianNoise(gauss_std[i])(x)  
    
    x = Dense(1, activation=None, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(x)

    model = Model(inpt, x)


    custom_loss = lambda y_t, y_p: qloss(y_t,y_p,q)
    model.compile(loss=custom_loss, optimizer='adadelta')
    model.fit(X, 
              Y, 
              epochs = 20, 
              batch_size = batch_size, 
              shuffle = True)


    if save_file is not None:
        model.save(save_file)

    return model


def predict(X, model_from=None, scale_par=None):

    model = load_model(model_from)

    predY = model.predict(X)

    if scale_par is not None: 
        logger.info('data is scaled, now mapping it back!')
        predY = predY*scale_par[target]['sigma'] + scale_par[target]['mu']

    return predY
        


def scale(df, scale_file):

    df = pd.DataFrame(df)
    try:
        par = pd.read_hdf(scale_file) 
    except FileNotFoundError:
        logger.info('File {} not found! Now run automatic scale! '.format(scale_file))
        par = pd.DataFrame([df.mean(), df.std()], index=['mu', 'sigma'])
        par.to_hdf(scale_file, key='scale_par', mode='w')

    df_scaled = (df - par.loc['mu',:])/par.loc['sigma',:]
    return df_scaled


def qloss(y_true, y_pred, q):
    e = (y_true - y_pred)
    return K.mean(K.maximum(q*e, (q-1.)*e), axis=-1)


