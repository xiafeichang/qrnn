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




def trainNN(X, Y, num_hidden_layers=3, num_units=None, act=None, sample_weight=None, l2lam=1.e-3, opt='SGD', lr=0.1, dp=None, gauss_std=None, batch_size=64, epochs=10, checkpoint_dir='./ckpt', save_file=None, evaluate_data=None, model_plot=None):

    # cleanup
    try:
        K.clear_session()
        print('Keras backend session cleared!')
    except:
        print('Failed to clear session, continue') 

    input_dim = len(X.keys())

    if len(Y.shape) == 1: 
        output_dim = 1
    else: 
        output_dim = Y.shape[1]
    
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
        model = get_compiled_model(input_dim, output_dim, num_hidden_layers, num_units, act, l2lam, opt, lr, dp, gauss_std)

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
        model.save(save_file)

    eval_results = None
    if evaluate_data is not None:
        eval_results = model.evaluate(evaluate_data[0], evaluate_data[1], batch_size=batch_size)

    if model_plot is not None:
        keras.utils.plot_model(model, to_file=model_plot, show_shapes=True)

    return history, eval_results


def predict(X, model_from=None, transformer=None, target_name=None):

    model = load_model(model_from)

    with tf.distribute.MirroredStrategy().scope():
        predY = model.predict(X)

#    print(predY)
    if transformer is not None: 
        print('target is transformed, now mapping it back!')
        if isinstance(target_name, list):
            predY = pd.DataFrame(predY, columns=target_name)
        predY = inverse_transform(predY, transformer, target_name) 

    return predY
        

def get_compiled_model(input_dim, output_dim, num_hidden_layers, num_units, act, l2lam=1.e-3, opt='SGD', lr=0.1, dp=None, gauss_std=None):

    inpt = Input((input_dim,), name='inpt')

    x = inpt
    
    for i in range(num_hidden_layers): 
        x = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(l2lam), 
                  activation=act[i])(x)
        if dp is not None:
            x = Dropout(dp[i])(x)
        if gauss_std is not None: 
            x = GaussianNoise(gauss_std[i])(x)  
    
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





