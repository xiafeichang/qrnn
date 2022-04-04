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


def trainQuantile(X, Y, qs, num_hidden_layers=1, num_units=None, act=None, qweights=None, dp=None, gauss_std=None, batch_size=64, epochs=10, checkpoint_dir='./ckpt', save_file=None):

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
#    strategy = tf.distribute.MirroredStrategy(devices= ["/gpu:0","/gpu:1"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = load_or_restore_model(checkpoint_dir, qs, input_dim, num_hidden_layers, num_units, act, qweights, dp, gauss_std)

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


def get_compiled_model(q, input_dim, num_hidden_layers, num_units, act, qweights, dp=None, gauss_std=None):
    
    inpt = Input((input_dim,), name='inpt')

    x = inpt
    
    for i in range(num_hidden_layers):
        x = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(0.001), activation=act[i])(x)
#        x = Dropout(dp[i])(x)
#        x = GaussianNoise(gauss_std[i])(x)  
    
    x = Dense(len(qweights), activation=None, use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal')(x)

    model = Model(inpt, x)

    def custom_loss(y_true, y_pred): 
        return qloss(y_true, y_pred, q, qweights)
    model.compile(loss=custom_loss, optimizer='adadelta')
    return model


def load_or_restore_model(checkpoint_dir, qs, input_dim, num_hidden_layers, num_units, act, qweights, dp=None, gauss_std=None):

    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        def custom_loss(y_true, y_pred): 
            return qloss(y_true, y_pred, qs, qweights)
        return load_model(latest_checkpoint, custom_objects={'custom_loss':custom_loss})
    print("Creating a new model")
    return get_compiled_model(qs, input_dim, num_hidden_layers, num_units, act, qweights, dp, gauss_std)


def qloss(y_true, y_pred, q, qweights):  # qweights is the weight of different variables
    qs = np.array([q for _ in qweights])
    qweight = np.array(qweights)
    e = y_true - y_pred

    # Hubber loss
    delta = 0.1
    is_small_e = K.abs(e) < delta
    small_e = K.square(e) / (2.*delta)
    big_e = K.abs(e) - delta/2.
    huber_e = K.sign(e)*tf.where(is_small_e, small_e, big_e) 

    return K.mean(K.maximum(q*huber_e, (q-1.)*huber_e)*qweight)
#    return K.mean(K.square(K.maximum(q*huber_e, (q-1.)*huber_e)*qweight))


