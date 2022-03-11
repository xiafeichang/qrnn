import argparse
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from qrnn import trainQuantile, predict, scale
from transformer import fit_transformer, transform, inverse_transform

from sklearn import preprocessing 
import pickle
import gzip

#from joblib import delayed, Parallel, parallel_backend, register_parallel_backend
#from dask.distributed import Client, LocalCluster, progress, wait, get_client
#from dask_jobqueue import SLURMCluster


def setup_cluster(config_file): 
    if config_file:
        if not isinstance(config_file, str):
            raise TypeError('if passed, config_file must be of type string')

        stream = open(config_file, 'r')
        inp = yaml.safe_load(stream)
        cores = inp['jobqueue']['slurm']['cores']
        memory = inp['jobqueue']['slurm']['memory']
        jobs = inp['jobqueue']['slurm']['jobs']
        walltime = inp['jobqueue']['slurm']['walltime']
        cluster = SLURMCluster(
                cores = cores,
                memory = memory,
                walltime = walltime
                )
        cluster.scale(jobs = jobs)
    else:
        cluster = LocalCluster()

    return cluster, Client(cluster)
 
def close_cluster(cluster, client): 
    client.close()
    cluster.close()

def gen_scale_par(df, variables, scale_file):
    df = df.loc[:,variables] 
    par = pd.DataFrame([df.mean(), df.std()], index=['mu', 'sigma'])
    par.to_hdf(scale_file, key='scale_par', mode='w')
    return par

def compute_qweights(sr, qs):
    quantiles = np.quantile(sr, qs)
    qweights = np.array([1./(quantiles[1]-quantiles[0])])
    for i in range(1,len(quantiles)-1):
        qweights = np.append(qweights, 2./(quantiles[i+1]-quantiles[i-1]))
    qweights = np.append(qweights, 1./(quantiles[-1]-quantiles[-2]))
    return qweights/np.min(qweights)

def test(X, Y, qs, qweights, model_from, scale_par=None, transformer=None): # transformer: tuple like, (transformer_file, variables)
    Y = np.array(Y)
    predY = predict(X, qs, qweights, model_from)
    es = (Y-predY.T).T
    loss = np.maximum(qs*es, (qs-1.)*es)*qweights
    if scale_par is not None:
        return np.mean(scale_par['sigma']*predY + scale_par['mu'],axis=0), loss
    elif transformer is not None:
        return np.mean(inverse_transform(predY, transformer[0], transformer[1]), axis=0), loss
    else: 
        return np.mean(predY,axis=0), loss
    

def main(options):
    variables = ['probeS4','probeR9','probeCovarianceIeIp','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 

    data_key = 'data'
    EBEE = 'EB'
      
    inputtrain = 'df_{}_{}_train.h5'.format(data_key, EBEE)
    inputtest = 'df_{}_{}_test.h5'.format(data_key, EBEE)
    
    #load dataframe
    nEvt = 5000000
    df_train = (pd.read_hdf(inputtrain).loc[:,kinrho+variables]).sample(nEvt, random_state=100).reset_index(drop=True)
    df_test_raw  = (pd.read_hdf(inputtest).loc[:,kinrho+variables]).sample(nEvt, random_state=100).reset_index(drop=True)
    
    # comments: good performence on smooth distribution, but not suitable for distributions with cutoffs
    num_hidden_layers = 5
    num_units = [2000, 2000, 2000, 1000, 500]
    act = ['tanh','exponential', 'softplus', 'tanh', 'elu']
    dropout = [0.1, 0.1, 0.1, 0.1, 0.1]
    gauss_std = [0.2, 0.2, 0.2, 0.2, 0.2]
    '''
    num_hidden_layers = 2
    num_units = [2000, 2000]
    act = ['tanh', 'softplus']
    dropout = [0.1, 0.1]
    gauss_std = [0.3, 0.3]
    '''

    #get or generate scale parameters

    scale_file = 'scale_para/data_{}_try.h5'.format(EBEE)
    scale_par = gen_scale_par(df_train, kinrho, scale_file)
#    try: 
#        scale_par = pd.read_hdf(scale_file)
#    except FileNotFoundError:
#        scale_par = gen_scale_par(df_train, kinrho+variables, scale_file)
    #scale or transform data
    transformer_file = '{}_{}'.format(data_key, EBEE)
    df_train.loc[:,variables] = transform(df_train.loc[:,variables], transformer_file, variables)
    df_test_raw.loc[:,variables] = transform(df_test_raw.loc[:,variables], transformer_file, variables)

    df_train.loc[:,kinrho] = scale(df_train.loc[:,kinrho], scale_file=scale_file)
    df_test_raw.loc[:,kinrho] = scale(df_test_raw.loc[:,kinrho], scale_file=scale_file)

    print(df_train)
    print(df_test_raw)

    #train
    
    train_start = time.time()

    target = variables[options.ith_var]
    features = kinrho + variables[:variables.index(target)] 
    print('>>>>>>>>> train for variable {} with features {}'.format(target, features))

    X = df_train.loc[:,features]
    Y = df_train.loc[:,target]
    
    qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
    qweights = np.ones_like(qs)
#    qweights = compute_qweights(Y, qs)
#    qweights = 0.25/(qs*np.flip(qs))
    print('quantile loss weights: {}'.format(qweights))

    trainQuantile(X, Y, qs, num_hidden_layers, num_units, act, qweights, dropout, gauss_std, batch_size = 8192, epochs=10, 
                  checkpoint_dir='ckpt/'+target, save_file = 'combined_models/{}_{}_{}'.format(data_key, EBEE, target))

    train_end = time.time()
    print('time spent in training: {} s'.format(train_end-train_start))
    
    #test
    matplotlib.use('agg')
    pT_scale_par = scale_par.loc[:,'probePt']
    pTs = (np.array([25., 30., 35., 40., 45., 50., 60., 150.]) - pT_scale_par['mu'])/pT_scale_par['sigma']
    for i in range(len(pTs)-1): 
        df_test = df_test_raw.query('probePt>' + str(pTs[i]) + ' and probePt<' + str(pTs[i+1]))
        X_test = df_test.loc[:,features]
        Y_test = df_test.loc[:,target]
     
        q_pred, loss_ = test(X_test, Y_test, qs, qweights, 
                             model_from='combined_models/{}_{}_{}'.format(data_key, EBEE, target)
                             )#, transformer=(transformer_file, target))
        if i==0:
            loss = loss_ 
        else: 
            loss = np.append(loss, loss_, axis=0)


        fig = plt.figure(tight_layout=True)
#        plt.hist(inverse_transform(df_test[target], transformer_file, target), bins=100, density=True, cumulative=True, histtype='step')
        plt.hist(df_test[target], bins=100, density=True, cumulative=True, histtype='step')
        plt.plot(q_pred, qs, 'o')
        fig.savefig('plots/combined_{}_{}_{}_{}.png'.format(data_key, EBEE, target, str(i)))
        plt.close(fig)
    
    print(loss.shape)
    loss_mean = np.mean(loss, axis=0)
    loss_std = np.std(loss, axis=0)
    print('mean: {}\nstd: {}'.format(loss_mean,loss_std))

 


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-i','--ith_var', action='store', type=int, required=True)
    options = parser.parse_args()
    main(options)
