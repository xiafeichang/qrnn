import yaml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from qrnn import QRNN

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
#    target = variables[1]
    
    pTs = [25., 30., 35., 40., 45., 50., 60., 150.]
    
    qs = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    num_hidden_layers = 3
    num_units = [1000, 500, 100]
    act = ['tanh', 'softplus', 'elu']
    
    for target in variables: 
        
        scale_para_file = 'scale_para/{}_{}.h5'.format(inputfile[3:-12], target)
        
        print('>>>>>>>>> train for variable {}'.format(target))
        qrnn = QRNN(df_train, features, target, scale_file=scale_para_file)
        
        cluster, client = setup_cluster('dask_cluster_config.yml')
        cluster.scale(len(qs))
        client.wait_for_workers(1)

        futures = [client.submit(
            qrnn.trainQuantile,
            q,
            num_hidden_layers,
            num_units,
            act,
            batch_size = 8192,
            save_file = 'models/{}_{}'.format(target,str(q).replace('.','p')),
            ) for q in qs ]

        results = client.gather(futures)

        close_cluster(cluster, client)
        del cluster 
        del client

if __name__ == "__main__": 
    main()
