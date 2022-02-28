import os
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


def trainQuantile(X, Y, num_hidden_layers=1, num_units=None, act=None, dp=None, gauss_std=None, batch_size=64, epochs=10, checkpoint_dir='./ckpt', save_file=None):

    input_dim = len(X.keys())
    
    if num_units is None: 
        num_units = [10 for i in range(num_hidden_layers)]
    
    if act is None:
        act = ['linear' for i in range(num_hidden_layers)]

    # check checkpoint dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # create a MirroredStrategy
    print('devices: ', tf.config.list_physical_devices('GPU'))
    strategy = tf.distribute.MirroredStrategy(devices= ["/gpu:0","/gpu:1"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = load_or_restore_model(checkpoint_dir, input_dim, num_hidden_layers, num_units, act, dp, gauss_std)

    # save checkpoint every epoch
    callbacks = [keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch")]

    model.fit(X, 
              Y, 
              epochs = epochs, 
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


def get_compiled_model(input_dim, num_hidden_layers, num_units, act, dp=None, gauss_std=None):

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
    return model


def load_or_restore_model(checkpoint_dir, input_dim, num_hidden_layers, num_units, act, dp=None, gauss_std=None):

    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return load_model(latest_checkpoint, custom_objects={'qloss':qloss})
    print("Creating a new model")
    return get_compiled_model(input_dim, num_hidden_layers, num_units, act, dp, gauss_std)


def qloss(y_true, y_pred):
    q = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    e = (y_true - y_pred)
    return K.mean(K.maximum(q*e, (q-1.)*e))


