import argparse
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn import preprocessing 
import pickle
import gzip

from qrnn import trainQuantile, predict, scale
from mylib.transformer import transform, inverse_transform



def gen_scale_par(df, variables, scale_file):
    df = df.loc[:,variables] 
    par = pd.DataFrame([df.mean(), df.std()], index=['mu', 'sigma'])
    par.to_hdf(scale_file, key='scale_par', mode='w')
    return par

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

def compute_qweights(sr, qs, weights=None):
    quantiles = np.quantile(sr, qs)
    es = np.array(sr)[:,None] - quantiles
    huber_e = Hubber(es, 1.e-4, signed=True)
    loss = np.maximum(qs*huber_e, (qs-1.)*huber_e)
    qweights = 1./np.average(loss, axis=0, weights=weights)
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
    variables = ['probePhiWidth','probeEtaWidth','probeSigmaIeIe','probeS4','probeR9','probeCovarianceIeIp']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    weight = ['ml_weight']

    data_key = options.data_key
    EBEE = options.EBEE 
     
    inputtrain = 'weighted_dfs/df_{}_{}_train.h5'.format(data_key, EBEE)
    inputtest = 'weighted_dfs/df_{}_{}_test.h5'.format(data_key, EBEE)
   
    #load dataframe
    nEvt = 3000000
    df_train = (pd.read_hdf(inputtrain).loc[:,kinrho+variables+weight]).sample(nEvt, random_state=100).reset_index(drop=True)
    df_test_raw  = (pd.read_hdf(inputtest).loc[:,kinrho+variables+weight]).sample(nEvt, random_state=100).reset_index(drop=True)
    
    #transform features and targets
    transformer_file = 'data_{}'.format(EBEE)
    df_train.loc[:,kinrho+variables] = transform(df_train.loc[:,kinrho+variables], transformer_file, kinrho+variables)
    df_test_raw.loc[:,kinrho+variables] = transform(df_test_raw.loc[:,kinrho+variables], transformer_file, kinrho+variables)

    print(df_train)
    print(df_test_raw)

    #train
    
    train_start = time.time()
    modeldir = 'chained_models'
    model_file_data = '{}/{}_{}_{}'.format(modeldir, 'data', EBEE, target)

    target = variables[options.ith_var]
    features = kinrho + variables[:variables.index(target)] 
#    features = kinrho + variables[:min(variables.index(target),3)] 
#    features = kinrho 
    print('>>>>>>>>> train for variable {} with features {}'.format(target, features))

    X = df_train.loc[:,features]
    Y = df_train.loc[:,target]
    sample_weight = df_train.loc[:,'ml_weight']
#    X_test_raw = df_test_raw.loc[:,features]
#    Y_test_raw = df_test_raw.loc[:,target]


    qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    qweights = compute_qweights(Y, qs, sample_weight)
    print('quantile loss weights: {}'.format(qweights))

    batch_size = pow(2, 13)
#    num_hidden_layers = 5
#    num_units_from = 50
#    shrink_rate = 0.8
#    num_units = [int((num_units_from*shrink_rate**i)/len(qs)) for i in range(num_hidden_layers)]
#    act = ['tanh' for _ in range(num_hidden_layers)]
#    dropout = [0.1 for _ in range(num_hidden_layers)]
#    gauss_std = None 

    num_hidden_layers = 6
    num_connected_layers = 3
    num_units = [30, 25, 30, 25, 20, 15]
    act = ['tanh' for _ in range(num_hidden_layers)]
#    act = ['tanh','exponential', 'softplus', 'elu', 'tanh']
    dropout = [0.1, 0.1, 0.1, 0.1, 0.1]
    gauss_std = [0.2, 0.2, 0.2, 0.2, 0.2]


#    train_dataset = tf.data.Dataset.from_tensor_slices((df_train[:,features], df_train[:,target])).batch(batch_size)
#    test_dataset = tf.data.Dataset.from_tensor_slices((df_train[:,features], df_train[:,target])).batch(batch_size)

    history, eval_results = trainQuantile(
        X, Y, 
        qs, qweights, 
        num_hidden_layers, num_units, act, 
        num_connected_layers = num_connected_layers,
        sample_weight = sample_weight,
        l2lam = 1.e-3, 
        opt = 'Adadelta', lr = 0.5, 
        batch_size = batch_size, 
        epochs = 1000, 
#        checkpoint_dir = 'ckpt/'+target, 
        save_file = model_file_data, 
        )

    train_end = time.time()
    print('evaluation results: ', eval_results)
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
    X_test = df_test_raw.loc[:,features]
    Y_test = df_test_raw.loc[:,target]
    
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
        df_test_raw['{}_pred_{}'.format(target, qs[i])] = np.array(predict(X_test, qs, qweights, model_from))[:,i]
     
        draw_plot(data_key, EBEE, df_test_raw, qs[i], pTs, '$p_T$', 'probePt', target, transformer_file)
        draw_plot(data_key, EBEE, df_test_raw, qs[i], etas, '$\eta$', 'probeScEta', target, transformer_file)
        draw_plot(data_key, EBEE, df_test_raw, qs[i], rhos, '$\\rho$', 'rho', target, transformer_file)
        draw_plot(data_key, EBEE, df_test_raw, qs[i], phis, '$\phi$', 'probePhi', target, transformer_file)


    #test
    pT_scale_par = scale_par.loc[:,'probePt']
    pTs = transform(np.array([25., 30., 35., 40., 45., 50., 60., 150.]), transformer_file, 'probePt')
    for i in range(len(pTs)-1): 
        df_test = df_test_raw.query('probePt>' + str(pTs[i]) + ' and probePt<' + str(pTs[i+1]))
        X_test = df_test.loc[:,features]
        Y_test = df_test.loc[:,target]
     
        q_pred, q_pred_err, loss_ = test(X_test, Y_test, qs, qweights, 
                                         model_from='models/{}_{}_{}'.format(data_key, EBEE, target)
                                         )#, transformer=(transformer_file, target))
        if i==0:
            loss = loss_ 
        else: 
            loss = np.append(loss, loss_, axis=0)


        fig = plt.figure(tight_layout=True)
#        plt.hist(inverse_transform(df_test[target], transformer_file, target), bins=100, density=True, cumulative=True, histtype='step')
        plt.hist(df_test[target], bins=100, density=True, cumulative=True, histtype='step', label='true distribution')
        plt.errorbar(q_pred, qs, xerr=q_pred_err, fmt='.', markersize=7, elinewidth=2, capsize=3, markeredgewidth=2, label='prediction')
        plt.xlabel(target)
        plt.legend()
        fig.savefig('plots/old_plots/{}_{}_{}_{}.png'.format(data_key, EBEE, target, str(i)))
        plt.close(fig)
    
    print(loss.shape)
    loss_mean = np.mean(loss, axis=0)
    loss_std = np.std(loss, axis=0)
    print('mean: {}\nstd: {}'.format(loss_mean,loss_std))

 


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-i','--ith_var', action='store', type=int, required=True)
    requiredArgs.add_argument('-d','--data_key', action='store', type=str, required=True)
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    options = parser.parse_args()
    main(options)
