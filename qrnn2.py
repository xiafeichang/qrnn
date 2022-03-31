import os
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Input, Dense, GaussianNoise, Dropout
from keras.models import Model, load_model
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
from tensorflow import keras
import tensorflow as tf


def trainQuantile(X, Y, qs, num_hidden_layers=1, num_units=None, act=None, qweights=None, dp=None, gauss_std=None, batch_size=64, epochs=10, checkpoint_dir='./ckpt', save_file=None):

    input_dim = len(X.keys())
    
    if num_units is None: 
        num_units = [10 for i in range(num_hidden_layers)]
    
    if act is None:
        act = ['linear' for i in range(num_hidden_layers)]

    if qweights is None: 
        qweights = np.ones_like(qs)

    # create a MirroredStrategy
    print('devices: ', tf.config.list_physical_devices('GPU'))
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = get_compiled_model(qs, input_dim, num_hidden_layers, num_units, act, qweights, dp, gauss_std)

    history = model.fit(
        X, Y, 
        epochs = epochs, 
        batch_size = batch_size, 
        validation_split = 0.1,
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, min_delta=0.00005, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=0.001, patience=3, verbose=1), 
            TerminateOnNaN()
            ], 
        shuffle = True,
        )


    if save_file is not None:
        model.save(save_file)

    return history


def predict(X, qs, qweights, model_from=None, scale_par=None):

    def custom_loss(y_true, y_pred): 
        return qloss(y_true, y_pred, qs, qweights)
    model = load_model(model_from, custom_objects={'custom_loss':custom_loss})

    with tf.distribute.MirroredStrategy().scope():
        predY = model.predict(X)

    if scale_par is not None: 
        predY = predY*scale_par['sigma'] + scale_par['mu']

    return predY
        


def scale(df, scale_file):

    df = pd.DataFrame(df)

    par = pd.read_hdf(scale_file).loc[:,df.keys()] 

    df_scaled = (df - par.loc['mu',:])/par.loc['sigma',:]
    return df_scaled


def get_compiled_model(qs, input_dim, num_hidden_layers, num_units, act, qweights, dp=None, gauss_std=None):
    
    inpt = Input((input_dim,), name='inpt')

    x = inpt
    
    for i in range(num_hidden_layers):
        x = Dense(
            num_units[i], 
            use_bias=True, 
            kernel_initializer='he_normal', 
            bias_initializer='he_normal',
            kernel_regularizer=regularizers.l2(1.e-3), 
            activation=act[i],
            )(x)
#        x = Dropout(dp[i])(x)
#        x = GaussianNoise(gauss_std[i])(x)  
    
    x = Dense(len(qs), activation='linear', use_bias=True, kernel_initializer=None, bias_initializer='he_normal')(x)

    model = Model(inpt, x)

    def custom_loss(y_true, y_pred): 
        return qloss(y_true, y_pred, qs, qweights)
    model.compile(loss=custom_loss, optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.e-1))
#    model.compile(loss=custom_loss, optimizer='adadelta')

    model.summary()
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

def qloss(y_true, y_pred, qs, qweights):
    q = np.array(qs)
    qweight = np.array(qweights)
    e = -(y_true - y_pred)
    return K.mean(K.maximum(q*e, (q-1.)*e)*qweight)
'''
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
'''


