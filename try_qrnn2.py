import argparse
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import pickle
import gzip

from qrnn2 import trainQuantile, predict, scale
from mylib.transformer import transform, inverse_transform


def gen_scale_par(df, variables, scale_file):
    df = df.loc[:,variables] 
    par = pd.DataFrame([df.mean(), df.std()], index=['mu', 'sigma'])
    par.to_hdf(scale_file, key='scale_par', mode='w')
    return par

def compute_qweights1(sr, qs):
    quantiles = np.quantile(sr, qs)
    qweights = np.array([1./(quantiles[1]-quantiles[0])])
    for i in range(1,len(quantiles)-1):
        qweights = np.append(qweights, 2./(quantiles[i+1]-quantiles[i-1]))
    qweights = np.append(qweights, 1./(quantiles[-1]-quantiles[-2]))
    return qweights/np.min(qweights)

def draw_plot(data_key, EBEE, df, q, x_vars, x_title, x_var_name, target, transformer_file):

    var_pred_mean = np.zeros(len(x_vars)-1)
    var_true_mean = np.zeros(len(x_vars)-1)
    x_vars_c = np.zeros(len(x_vars)-1)
    for i in range(len(x_vars)-1):
        query_str = x_var_name + ' > ' + str(x_vars[i]) + ' and ' + x_var_name +' < ' + str(x_vars[i+1])

#        var_pred_mean[i] = np.mean(inverse_transform((df.query(query_str))[f'{target}_pred'], transformer_file, target))
#        var_true_mean[i] = np.mean(inverse_transform((df.query(query_str))[target], transformer_file, target))

        var_pred_mean[i] = np.mean((df.query(query_str))[f'{target}_pred_{q}'])
        var_true_mean[i] = np.quantile((df.query(query_str))[target], q)

        x_vars_c[i] = ((x_vars[i] + x_vars[i+1])/2.)
    
    x_vars_c = inverse_transform(x_vars_c, transformer_file, x_var_name)

    fig = plt.figure(tight_layout=True)
    plt.plot(x_vars_c, var_pred_mean, label='predicted')
    plt.plot(x_vars_c, var_true_mean, label='true')
    plt.xlabel(x_title)
    plt.ylabel('mean of {}'.format(target))
    plt.legend()
    fig.savefig('plots/check_results/{}_{}_{}_{}_{}.png'.format(data_key, EBEE, target, x_var_name, q))
    plt.close(fig)

def compute_qweights(sr, qs):
    quantiles = np.quantile(sr, qs)
    es = np.array(sr)[:,None] - quantiles
    huber_e = Hubber(es, 1.e-4, signed=True)
    loss = np.maximum(qs*huber_e, (qs-1.)*huber_e)
    print('ideal loss: ', np.mean(loss, axis=0))
    qweights = 1./np.mean(loss, axis=0)
    return qweights/np.min(qweights)

def Hubber(e, delta=0.1, signed=False):
    is_small_e = np.abs(e) < delta
    small_e = np.square(e) / (2.*delta)
    big_e = np.abs(e) - delta/2.
    if signed:
        return np.sign(e)*np.where(is_small_e, small_e, big_e) 
    else: 
        return np.where(is_small_e, small_e, big_e)

def test(X, Y, qs, qweights, model_from, scale_par=None, transformer=None): # transformer: tuple like, (transformer_file, variables)
    Y = np.array(Y)
    predY = predict(X, qs, qweights, model_from)
    es = (Y-predY.T).T
    huber_e = Hubber(es, 1.e-4, signed=True)
    loss = np.maximum(qs*huber_e, (qs-1.)*huber_e)*qweights
    if scale_par is not None:
        return np.mean(scale_par['sigma']*predY + scale_par['mu'],axis=0), loss
    elif transformer is not None:
        return np.mean(inverse_transform(predY, transformer[0], transformer[1]), axis=0), loss
    else: 
        return np.mean(predY,axis=0), np.std(predY,axis=0), loss
     

def main(options):
    variables = ['probeS4','probeR9','probeCovarianceIeIp','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    weight = ['ml_weight']

    data_key = 'mc'
    EBEE = 'EB'
      
    inputtrain = 'weighted_dfs/df_{}_{}_train.h5'.format(data_key, EBEE)
    inputtest = 'weighted_dfs/df_{}_{}_test.h5'.format(data_key, EBEE)
#    inputtest = inputtrain
    
    #load dataframe
    nEvt = 1000000
    df_train = (pd.read_hdf(inputtrain).loc[:,kinrho+variables+weight]).sample(nEvt, random_state=100).reset_index(drop=True)
    df_test  = (pd.read_hdf(inputtest).loc[:,kinrho+variables+weight]).sample(nEvt, random_state=100).reset_index(drop=True)

    # comments: good performence on smooth distribution, but not suitable for distributions with cutoffs
    '''
    num_hidden_layers = 5
    num_units = [2000, 1000, 500, 200, 100]
    act = ['tanh','exponential', 'softplus', 'tanh', 'elu']
#    num_units = [2000 for _ in range(num_hidden_layers)]
#    act = ['tanh' for _ in range(num_hidden_layers)]
    dropout = [0.1, 0.1, 0.1, 0.1, 0.1]
    gauss_std = [0.2, 0.2, 0.2, 0.2, 0.2]
    '''
    num_hidden_layers = 5
    num_units = [200 for _ in range(num_hidden_layers)]
    act = ['tanh' for _ in range(num_hidden_layers)]
    dropout = [0.1, 0.1]
    gauss_std = [0.3, 0.3]

    #get or generate scale parameters
    transformer_file = 'data_{}'.format(EBEE)
    df_train = transform(df_train, transformer_file, kinrho+variables)
    df_test = transform(df_test, transformer_file, kinrho+variables)

    #train
    
    train_start = time.time()

    target = variables[options.ith_var]
    features = kinrho# + variables[:variables.index(target)] 
    print('>>>>>>>>> train for variable {} with features {}'.format(target, features))

    X = df_train.loc[:,features]
    Y = df_train.loc[:,target]
    sample_weight = df_train.loc[:,'ml_weight']
    
    qs = np.array([0.5])
#    qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
#    qs = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
#    qweights = np.ones_like(qs)
    qweights = compute_qweights(Y, qs)
#    qweights = np.power(0.25/(qs*np.flip(qs)),1)
    print('quantile loss weights: {}'.format(qweights))

    history = trainQuantile(
        X, Y, qs, num_hidden_layers, num_units, act, qweights, 
        sample_weight = sample_weight,
        dp = dropout, gauss_std = gauss_std, 
        batch_size = 1024, epochs=1000, 
#        checkpoint_dir='ckpt/'+target, 
        save_file = 'combined_models/{}_{}_{}'.format(data_key, EBEE, target),
        )

    train_end = time.time()
    print('time spent in training: {} s'.format(train_end-train_start))

    matplotlib.use('agg')
    # plot training history
    history_fig = plt.figure(tight_layout=True)
    plt.plot(history.history['loss'], label='training')
    plt.plot(history.history['val_loss'], label='validation')
    plt.yscale('log')
    plt.title('Training history')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    history_fig.savefig('plots/training_histories/{}_{}_{}.png'.format(data_key, EBEE, target))


    # check result
    X_test = df_test.loc[:,features]
    Y_test = df_test.loc[:,target]
    
    #pTs_raw = np.array([25., 30., 32.5, 35., 37.5, 40., 42.5, 45., 50., 60., 150.])
    pTs_raw = np.arange(25., 55., 1.5)
    pTs = transform(pTs_raw, transformer_file, 'probePt')
    etas_raw = np.arange(-1.45, 1.45, 0.15)
    etas = transform(etas_raw, transformer_file, 'probeScEta')
    #rhos_raw = np.array([0., 8., 12., 15., 18., 21., 24., 27., 30., 36., 60.])
    rhos_raw = np.arange(0., 50., 2.)
    rhos = transform(rhos_raw, transformer_file, 'rho')
    phis_raw = np.arange(-3.15, 3.15, 0.3)
    phis = transform(phis_raw, transformer_file, 'probePhi')
    
    matplotlib.use('agg')
    
    model_from = 'combined_models/{}_{}_{}'.format(data_key, EBEE, target)
    for i in range(len(qs)): 
        df_test['{}_pred_{}'.format(target, qs[i])] = np.array(predict(X_test, qs, qweights, model_from))[:,i]
     
        draw_plot(data_key, EBEE, df_test, qs[i], pTs, '$p_T$', 'probePt', target, transformer_file)
        draw_plot(data_key, EBEE, df_test, qs[i], etas, '$\eta$', 'probeScEta', target, transformer_file)
        draw_plot(data_key, EBEE, df_test, qs[i], rhos, '$\\rho$', 'rho', target, transformer_file)
        draw_plot(data_key, EBEE, df_test, qs[i], phis, '$\phi$', 'probePhi', target, transformer_file)


 


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-i','--ith_var', action='store', type=int, required=True)
#    requiredArgs.add_argument('-q','--quantile', action='store', type=float, required=True)
    options = parser.parse_args()
    main(options)
