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

class QRNN(object):
    """
    class to perform quantile regression with a neural network 

            df: pandas dataframe, contains both freatures and target
      features: array like, contains names of the features to be used
        target: string, the name of the target
    scale_para: string, name of the hdf file contains the scale parameters, if not exist, run automatic scaling and create this file 
     use_model: string, name of the existing model 

    """
    

    def __init__(self, df, features, target, scale_file=None, use_model=None, quantile=None):
        
        self.features = features
        self.target = target

        if scale_file is not None:
            self.scale = True
            self.scale_par, df_scaled = scale(df, features+[target], scale_file)
            self.X = df_scaled.loc[:,self.features]
            self.Y = df_scaled.loc[:,self.target]

        else:
            self.scale = False
            self.X = df.loc[:,self.features]
            self.Y = df.loc[:,self.target]

'''
        if use_model is not None:
            self.model = load_model(use_model, compile=False) 
            if quantile is not None: 
                self.q = quantile
            else: 
                raise TypeError("To use an existing model, specify the quantile with 'quantile=your_quantile'") 
'''


    def trainQuantile(self, q, num_hidden_layers=1, num_units=None, act=None, batch_size=64, save_file=None):

        input_dim = len(self.features)
        
        if num_units is None: 
            num_units = [10 for i in range(num_hidden_layers)]
        
        if act is None:
            act = ['linear' for i in range(num_hidden_layers)]
    

        inpt = Input((input_dim,), name='inpt')
        
        x = inpt
        
        for i in range(num_hidden_layers):
            x = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.001), activation=act[i])(x)
#            x = Dropout(dp[i])(x)
#            x = GaussianNoise(gauss_std[i])(x)  
        
        x = Dense(1, activation=None, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(x)
    
        self.model = Model(inpt, x)
 

        custom_loss = lambda y_t, y_p: qloss(y_t,y_p,q)
        self.model.compile(loss=custom_loss, optimizer='adadelta')
        self.model.fit(self.X, 
                       self.Y, 
                       epochs = 20, 
                       batch_size = batch_size, 
                       shuffle = True)


        if save_file is not None:
            self.model.save(save_file)

        return self.model
    

    def predict(self, model_from=None):

        try:
            model = load_model(model_from)
        except FileNotFoundError: 
            model = self.model

        self.predictedY = model.predict(self.X)

        if self.scale: 
            logger.info('data is scaled, now mapping it back!')
            self.predictedY = self.predictedY*self.scale_par[self.target]['sigma'] + self.scale_par[self.target]['mu']

        return self.predictedY
        


def scale(df, variables, scale_file):

    df_ = df.loc[:,variables]
    try:
        par = pd.read_hdf(scale_file) 
    except FileNotFoundError:
        logger.info('File {} not found! Now run automatic scale! '.format(scale_file))
        par = pd.DataFrame([df_.mean(), df_.std()], index=['mu', 'sigma'])
        par.to_hdf(scale_file, key='scale_par', mode='w')

    df_scaled = (df_ - par.loc['mu',:])/par.loc['sigma',:]
    return par, df_scaled


def qloss(y_true, y_pred, q):
    e = (y_true - y_pred)
    return K.mean(K.maximum(q*e, (q-1.)*e), axis=-1)


