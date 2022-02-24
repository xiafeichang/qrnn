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


def trainQuantile(X, Y, num_hidden_layers=1, num_units=None, act=None, batch_size=64, save_file=None):

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
#        x = Dropout(dp[i])(x)
#        x = GaussianNoise(gauss_std[i])(x)  
    
    x = Dense(21, activation=None, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(x)

    model = Model(inpt, x)


    model.compile(loss=qloss, optimizer='adadelta')
    model.fit(X, 
              Y, 
              epochs = 10, 
              batch_size = batch_size, 
              shuffle = True)


    if save_file is not None:
        model.save(save_file)

    return 0
#    return model


def predict(X, model_from=None, scale_par=None):

    model = load_model(model_from, custom_objects={'qloss':qloss})

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


def qloss(y_true, y_pred):
    q = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    e = (y_true - y_pred)
    return K.mean(K.maximum(q*e, (q-1.)*e))


