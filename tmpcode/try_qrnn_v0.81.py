import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from qrnn import trainQuantile, predict, scale

#from joblib import delayed, Parallel, parallel_backend, register_parallel_backend
from dask.distributed import Client, LocalCluster, progress, wait, get_client
from dask_jobqueue import SLURMCluster


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

def test(X, model_from, scale_par):
    return np.mean(predict(X, model_from, scale_par),axis=0)
    

def main():
    variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 

    data_key = 'data'
    EBEE = 'EB'
      
    inputtrain = 'df_{}_{}_train.h5'.format(data_key, EBEE)
    inputtest = 'df_{}_{}_test.h5'.format(data_key, EBEE)
    
    #load dataframe
    df_train = (pd.read_hdf(inputtrain).loc[:,kinrho+variables])#.sample(100, random_state=100).reset_index(drop=True)
    df_test_raw  = (pd.read_hdf(inputtest).loc[:,kinrho+variables])#.sample(100, random_state=100).reset_index(drop=True)
    
    # comments: good performence on smooth distribution, but not suitable for distributions with cutoffs
    '''
    num_hidden_layers = 5
    num_units = [2000, 1000, 500, 200, 100]
    act = ['tanh','exponential', 'softplus', 'tanh', 'elu']
    dropout = [0.1, 0.1, 0.1, 0.1, 0.1]
    gauss_std = [0.2, 0.2, 0.2, 0.2, 0.2]
    '''
    num_hidden_layers = 2
    num_units = [2000, 2000]
    act = ['tanh', 'softplus']
    dropout = [0.1, 0.1]
    gauss_std = [0.3, 0.3]

    #get or generate scale parameters
    scale_file = 'scale_para/data_{}.h5'.format(EBEE)
    try: 
        scale_par = pd.read_hdf(scale_file)
    except FileNotFoundError:
        scale_par = gen_scale_par(df_train, kinrho+variables, scale_file)
    #scale data
    df_train = scale(df_train, scale_file=scale_file)
    df_test_raw = scale(df_test_raw, scale_file=scale_file)
        
    #train
    
    train_start = time.time()

    target = variables[5]
    features = kinrho + variables[:variables.index(target)] 
    print('>>>>>>>>> train for variable {} with features {}'.format(target, features))

    X = df_train.loc[:,features]
    Y = df_train.loc[:,target]
    
    trainQuantile(X, Y, num_hidden_layers, num_units, act, dropout, gauss_std, batch_size = 8192, epochs=10, 
                  checkpoint_dir='ckpt/'+target, save_file = 'combined_models/{}_{}_{}'.format(data_key, EBEE, target))

    train_end = time.time()
    print('time spent in training: {} s'.format(train_end-train_start))
    
    #test
#    plt.ioff()
    matplotlib.use('agg')
    target_scale_par = scale_par.loc[:,target]
    pT_scale_par = scale_par.loc[:,'probePt']
    pTs = (np.array([25., 30., 35., 40., 45., 50., 60., 150.]) - pT_scale_par['mu'])/pT_scale_par['sigma']
    for i in range(len(pTs)-1): 
        df_test = df_test_raw.query('probePt>' + str(pTs[i]) + ' and probePt<' + str(pTs[i+1]))
        X_test = df_test.loc[:,features]
     
        q_pred = test(X_test, model_from='combined_models/{}_{}_{}'.format(data_key, EBEE, target), scale_par=target_scale_par)


        qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
        fig = plt.figure(tight_layout=True)
        plt.hist((df_test[target]*target_scale_par['sigma']+target_scale_par['mu']), bins=100, density=True, cumulative=True, histtype='step')
        plt.plot(q_pred, qs, 'o')
        fig.savefig('plots/combined_{}_{}_{}_{}.png'.format(data_key, EBEE, target, str(i)))
#        fig.savefig('plots/combined_{}_{}_{}_{}.pdf'.format(data_key, EBEE, target, str(i)))
        plt.close(fig)

 


if __name__ == "__main__": 
    main()
