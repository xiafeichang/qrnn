import yaml
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from qrnn import trainQuantile, predict

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

def test(X, q, model_from, scale_par, target):
    return q, np.mean(predict(X, model_from, scale_par, target))
    

def main():
    variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
      
    inputfile = 'df_data_EB_train.h5'
    n_evt = 2000000
    
    #load dataframe
    df_total = pd.read_hdf(inputfile)
    df_smp = df_total.sample(n_evt, random_state=100).reset_index(drop=True)
    
    df_train = df_smp[:1000000] 
    df_test_raw  = df_smp[1000000:] 
    
    #set features and target
    features = kinrho 
    target = variables[0]
    
    pTs = [25., 30., 35., 40., 45., 50., 60., 150.]
    
    qs = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    num_hidden_layers = 5
    num_units = [500, 300, 200, 100, 50]
    act = ['tanh','exponential', 'softplus', 'tanh', 'elu']

    #generate scale parameters
    scale_par = 'scale_para/{}.h5'.format(inputfile[3:-12])
    gen_scale_par(df_train, kinrho+variables, scale_par)
        
    #train
    #for target in variables: 
    
    print('>>>>>>>>> train for variable {}'.format(target))
    X = df_train.loc[:,features]
    Y = df_train.loc[:,target]
    
    cluster, client = setup_cluster('dask_cluster_config.yml')
    cluster.scale(len(qs))
    client.wait_for_workers(1)

    futures = [client.submit(trainQuantile,
                             X, 
                             Y, 
                             q,
                             num_hidden_layers,
                             num_units,
                             act,
                             batch_size = 8192,
                             scale_par = scale_par,
                             save_file = 'models/{}_{}'.format(target,str(q).replace('.','p'))
                             ) for q in qs ]

    results = client.gather(futures)
    del futures

    close_cluster(cluster, client)
    del cluster 
    del client
    
    #test
#    plt.ioff()
    matplotlib.use('agg')
    print('>>>>>>>>> test for variable {}'.format(target))
    pTs = [25., 30., 35., 40., 45., 50., 60., 150.]
    for i in range(len(pTs)-1): 
        df_test = df_test_raw.query('probePt>' + str(pTs[i]) + ' and probePt<' + str(pTs[i+1]))
        X_test = df_test.loc[:,features]
     
        cluster, client = setup_cluster('dask_cluster_config.yml')
        cluster.scale(len(qs))
        client.wait_for_workers(1)

        futures_test = [client.submit(test,
                                      X_test, 
                                      q,
                                      model_from = 'models/{}_{}'.format(target,str(q).replace('.','p')),
                                      scale_par = scale_par, 
                                      target = target
                                      ) for q in qs ]

        progress(futures_test)
        test_results = np.array(client.gather(futures_test)).T

        fig = plt.figure(tight_layout=True)
        plt.hist(df_test[target], bins=100, density=True, cumulative=True, histtype='step')
        plt.plot(test_results[1], test_results[0], 'o')
        fig.savefig('plots/' + target + '_' + str(i) + '.png')
#        fig.savefig('plots/' + target + '_' + str(i) + '.pdf')
        plt.close(fig)

        close_cluster(cluster, client)
        del cluster 
        del client

 


if __name__ == "__main__": 
    main()
