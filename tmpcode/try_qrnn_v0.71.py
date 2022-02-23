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

def test(X, model_from, scale_par):
    return np.mean(predict(X, model_from, scale_par),axis=0)
    

def main():
    variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
      
    inputtrain = 'df_data_EB_train.h5'
    inputtest = 'df_data_EB_test.h5'
    
    #load dataframe
    df_train = (pd.read_hdf(inputtrain).loc[:,kinrho+variables])
    df_test_raw  = (pd.read_hdf(inputtest).loc[:,kinrho+variables])#.sample(1000, random_state=100).reset_index(drop=True)
    
    '''
    # comments: good performence on smooth distribution, but not suitable for distributions with cutoffs
    num_hidden_layers = 5
    num_units = [500, 300, 200, 100, 50]
    act = ['tanh','exponential', 'softplus', 'tanh', 'elu']
    '''

    num_hidden_layers = 3
    num_units = [1000, 300, 100]
    act = ['tanh','exponential', 'softplus']

    #generate scale parameters
    scale_par = 'scale_para/{}.h5'.format(inputtrain[3:-12])
    gen_scale_par(df_train, kinrho+variables, scale_par)
    #scale data
    df_train = scale(df_train, scale_file=scale_par)
    df_test_raw = scale(df_test_raw, scale_file=scale_par)
        
    #train
    
    train_start = time.time()

    features = kinrho 
#    target = variables[0]
#    for target in variables: 
#        features = kinrho + variables[:variables.index(target)]
#        print('>>>>>>>>> train for variable {} with features {}'.format(target, features))

    X = df_train.loc[:,features]
    Y = df_train.loc[:,target]
    
    # setup clusters with dask
    cluster, client = setup_cluster('dask_cluster_config.yml')
    cluster.scale(6)
    client.wait_for_workers(1)

    futures = [client.submit(trainQuantile,
                             X, 
                             Y, 
                             num_hidden_layers,
                             num_units,
                             act,
                             batch_size = 8192,
                             save_file = 'combined_models/{}'.format(target)
                             ) for target in variables ]

    results = client.gather(futures)
    del futures

    close_cluster(cluster, client)
    del cluster 
    del client
    
    train_end = time.time()
    print('time spent in training: {} s'.format(train_end-train_start))
    
    #test
#    plt.ioff()
    matplotlib.use('agg')
    pT_scale_par = pd.read_hdf(scale_par).loc[:,'probePt']
    pTs = (np.array([25., 30., 35., 40., 45., 50., 60., 150.]) - pT_scale_par['mu'])/pT_scale_par['sigma']
    for i in range(len(pTs)-1): 
        df_test = df_test_raw.query('probePt>' + str(pTs[i]) + ' and probePt<' + str(pTs[i+1]))
        X_test = df_test.loc[:,features]
     
        cluster, client = setup_cluster('dask_cluster_config.yml')
        cluster.scale(len(qs))
        client.wait_for_workers(1)

#        for target in variables: 
#            print('>>>>>>>>> test for variable {}'.format(target))
#            target_scale_par = pd.read_hdf(scale_par).loc[:,target]
        futures_test = [client.submit(test,
                                      X_test, 
                                      model_from = 'combined_models/{}'.format(target),
                                      scale_par = pd.read_hdf(scale_par).loc[:,target], 
                                      ) for target in variables ]

        progress(futures_test)
        test_results = np.array(client.gather(futures_test)).T

        fig = plt.figure(tight_layout=True)
        plt.hist((df_test[target]*target_scale_par['sigma']+target_scale_par['mu']), bins=100, density=True, cumulative=True, histtype='step')
        plt.plot(test_results[1], test_results[0], 'o')
        fig.savefig('plots/' + target + '_' + str(i) + '.png')
#        fig.savefig('plots/' + target + '_' + str(i) + '.pdf')
        plt.close(fig)

        close_cluster(cluster, client)
        del cluster 
        del client

 


if __name__ == "__main__": 
    main()
