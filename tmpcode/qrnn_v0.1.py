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
    

    def __init__(self, df, features, target, scale_para=None, use_model=None, quantile=None):
        
        self.features = features
        self.target = target
        self.data = df
        self.X = df.loc[:,features]
        self.Y = df.loc[:,target]

        if scale_para is not None:
            try:
                self._scale_from_file(scale_para)
            except FileNotFoundError:
                logger.info('File {} not found! Now run automatic scale! '.format(scale_para))
                self.scaleFeatures(self.features, save_file=scale_para)

        if use_model is not None:
            self.model = load_model(use_model, compile=False) 
            if quantile is not None: 
                self.q = quantile
            else: 
                raise TypeError("To use an existing model, specify the quantile with 'quantile=your_quantile'") 


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
            self.model.save(save_file, save_format='h5')

        return self.model
    

    def scaleFeatures(self, variables, save_file, scale_target=False):

        logger.info('scaling variables: {}'.format(variables))
        scale_par = pd.DataFrame(index=['mu','sigma'])

        for key in variables:
            self.X['{}_scaled'.format(key)], mu, sigma = scale(self.X[key], data_type=data_type(key))
            scale_par[key] = [mu, sigma]

        self.scaled_features = []
        for i in range(len(self.features)):
            if self.features[i] in variables: 
                self.scaled_features.append('{}_scaled'.format(self.features[i]))

        self.X = self.X.loc[:,self.scaled_features]
        print(self.X.keys())

        if (scale_target):
            logger.info("scaling target: '{}'".format(self.target))
            self.Y, self.Ymu, self.Ysigma = scale(self.Y, data_type(self.target))
            self.Y = self.Y.rename(self.target+'_scaled')
            scale_par[self.target] = [self.Ymu, self.Ysigma]

        scale_par.to_hdf(save_file, key='scale_par', mode='w')


    def predict(self):

        self.model.compile(loss=lambda y_t, y_p: qloss(y_t,y_p,self.q), optimizer='adadelta')
        self.predictedY = self.model.predict(self.X)
        if self.Y.name.endswith('_scaled'): 
            logger.info('target is scaled, now mapping it back!')
            self.predictedY = self.predictedY*self.Ysigma + self.Ymu

        return self.predictedY
        


    def _scale_from_file(self, scale_para):
        
        scale_par = pd.read_hdf(scale_para)
        for key in scale_par.keys():
            if key in self.features:
                logger.info("scaling feature '{}' from file {}".format(key, scale_para))
                self.X['{}_scaled'.format(key)] = scale(self.X[key], mu=scale_par[key]['mu'], sigma=scale_par[key]['sigma'])

            elif key == self.target: 
                logger.info("scaling target: '{}' from file {}".format(key, scale_para))
                self.Ymu = scale_par[key]['mu']
                self.Ysigma = scale_par[key]['sigma']
                self.Y = scale(self.Y, mu=self.Ymu, sigma=self.Ysigma)
                self.Y = self.Y.rename(self.target+'_scaled')

        self.scaled_features = []
        for i in range(len(self.features)):
            if self.features[i] in scale_par.keys(): 
                self.scaled_features.append('{}_scaled'.format(self.features[i]))

        self.X = self.X.loc[:,self.features]


   

def data_type(key):
    
    if key in ['rho']:
        return 'Poission'
    else: 
        return 'Gaussian'

def scale(a, data_type=None, mu=None, sigma=None):

    if data_type is not None: 
        mu = np.nanmean(a)
        if data_type == 'Gaussian':
            sigma = np.nanstd(a)
            a_scaled = (a - mu)/sigma
        elif data_type == 'Poission':
            sigma = np.nan
            a_scaled = a/mu
        return a_scaled, mu, sigma

    elif (mu is not None) and (sigma is not None): 
        if np.isnan(sigma): 
            a_scaled = a/mu
        else: 
            a_scaled = (a-mu)/sigma
        return a_scaled

    else: 
        raise TypeError("Wrong arguments are given: scale() requires either 'data_type' or both 'mu' and 'sigma' to be passed into!")


def qloss(y_true, y_pred, q):
    e = (y_true - y_pred)
    return K.mean(K.maximum(q*e, (q-1.)*e), axis=-1)


