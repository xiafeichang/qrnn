import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Input, Dense, GaussianNoise, Dropout
from keras.models import Model
from keras import regularizers
from tensorflow import keras
import tensorflow as tf

class QRNN(object):
    

    def __init__(self, df, features, target):
        
        self.features = features
        self.target = target
        self.data = df
        self.X = df.loc[:,features]
        self.Y = df.loc[:,target]
    

    def trainQuantile(self, q, num_hidden_layers=1, num_units=None, act=None, batch_size=64):

        input_dim = len(self.features)
        
        if num_units is None: 
            num_units = [10 for i in range(num_hidden_layers)]
        
        if act is None:
            act = ['linear' for i in range(num_hidden_layers)]
    
        self.model = self._get_model(input_dim, num_units, act, num_hidden_layers)
    
        self.model.compile(loss=lambda y_t, y_p: self._qloss(y_t,y_p,q), optimizer='adadelta')
        self.model.fit(self.X, 
                       self.Y, 
                       epochs = 10, 
                       batch_size = batch_size, 
                       shuffle = True)

        return self.model
    

    def scaleFeatures(self, features, save_file):

        scale_par = pd.DataFrame(index=['mu','sigma'])
        for key in features:
            if key in ['rho']:
                data_type = 'Poission'
            else: 
                data_type = 'Gaussian'
    
            self.X['{}_scaled'.format(key)], mu, sigma = self._scale(self.X[key], data_type)
            scale_par[key] = [mu, sigma]

        scale_par.to_hdf(save_file, key='scale_par', mode='w')
    
        for i in range(len(self.features)):
            if self.features[i] in features: 
                self.features[i] = self.features[i] + '_scaled'

        self.X = self.X.loc[:,self.features]

        return self.features


    def saveModel(self, fileName):

        self.model.save(fileName)
    

    def _scale(self, a, data_type):
        mu = np.nanmean(a)
        if data_type == 'Gaussian':
            sigma = np.nanstd(a)
            a_scaled = (a - mu)/sigma
        elif data_type == 'Poission':
            sigma = np.nan
            a_scaled = a/mu
    
        return a_scaled, mu, sigma


    def _qloss(self, y_true, y_pred, q):
        e = (y_true - y_pred)
        return K.mean(K.maximum(q*e, (q-1.)*e), axis=-1)


    def _get_model(self, input_dim, num_units, act, dp=0.1, gauss_std=0.3, num_hidden_layers=1):
    
        inpt = Input((input_dim,), name='inpt')
        
        x = inpt
        
        for i in range(num_hidden_layers):
            x = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.001), activation=act[i])(x)
#            x = Dropout(dp[i])(x)
#            x = GaussianNoise(gauss_std[i])(x)  
        
        x = Dense(1, activation=None, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(x)
    
        model = Model(inpt, x)
        return model
 

