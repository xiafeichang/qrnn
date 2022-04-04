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


def gen_scale_par(df, variables, scale_file):
    df = df.loc[:,variables] 
    par = pd.DataFrame([df.mean(), df.std()], index=['mu', 'sigma'])
    par.to_hdf(scale_file, key='scale_par', mode='w')

def test(X, q, model_from, scale_par):
    return q, np.mean(predict(X, model_from, scale_par))
    

def main():
    variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
      
    inputtrain = 'df_data_EB_train.h5'
    inputtest = 'df_data_EB_test.h5'
    
    #load dataframe
    df_train = (pd.read_hdf(inputtrain).loc[:,kinrho+variables])#.sample(1000, random_state=100).reset_index(drop=True)
    df_test_raw  = (pd.read_hdf(inputtest).loc[:,kinrho+variables])#.sample(1000, random_state=100).reset_index(drop=True)
    
    #set features and target
    features = kinrho 
#    target = variables[0]
    
    pTs = [25., 30., 35., 40., 45., 50., 60., 150.]
    
    qs = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

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
    for q in qs: 
        print('>>>>>>>>> train for variable {} with features {}'.format(target, features))

        X = df_train.loc[:,features]
        Y = df_train.loc[:,target]
        
        futures = [client.submit(trainQuantile,
                                 X, 
                                 Y, 
                                 q,
                                 num_hidden_layers,
                                 num_units,
                                 act,
                                 batch_size = 8192,
                                 save_file = 'models/{}_{}'.format(target,str(q).replace('.','p'))
                                 ) for q in qs ]

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
     
        for target in variables: 
            print('>>>>>>>>> test for variable {}'.format(target))
            target_scale_par = pd.read_hdf(scale_par).loc[:,target]
            futures_test = [client.submit(test,
                                          X_test, 
                                          q,
                                          model_from = 'models/{}_{}'.format(target,str(q).replace('.','p')),
                                          scale_par = target_scale_par, 
                                          ) for q in qs ]

            test_results = np.array(client.gather(futures_test)).T

            fig = plt.figure(tight_layout=True)
            plt.hist((df_test[target]*target_scale_par['sigma']+target_scale_par['mu']), bins=100, density=True, cumulative=True, histtype='step')
            plt.plot(test_results[1], test_results[0], 'o')
            fig.savefig('plots/' + target + '_' + str(i) + '.png')
#            fig.savefig('plots/' + target + '_' + str(i) + '.pdf')
            plt.close(fig)

 


if __name__ == "__main__": 
    main()
