import argparse
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from qrnn import trainQuantile, predict, scale

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

def get_qweights(target):
    qwsIdf = pd.DataFrame({'probeS4':[0.03588478, 0.13480885, 0.22564455, 0.28340023, 0.32600966, 0.35247085, 0.37060138, 0.37293005, 0.37105685, 0.36551802, 0.35261605, 0.33525826, 0.31252958, 0.28353584, 0.25343311, 0.22196003, 0.18607529, 0.14865117, 0.10640332, 0.05916152, 0.01610501],
                           'probeR9':[0.02670744, 0.09379196, 0.14226358, 0.17759852, 0.1998296, 0.21719939, 0.23120884, 0.23185595, 0.23509117, 0.23585299, 0.23199268, 0.2218315, 0.21316461, 0.19947866, 0.1875615, 0.16885376, 0.14483574, 0.12583409, 0.09132041, 0.05714136, 0.01692484],
                           'probeCovarianceIeIp':[0.04406181, 0.11396861, 0.17054979, 0.21213174, 0.24405916, 0.27311447, 0.29075913, 0.3050546 , 0.31668225, 0.32712497, 0.32353601, 0.32319342, 0.32099774, 0.30544753, 0.29101529, 0.26815126, 0.24117485, 0.20772284, 0.16658191, 0.1120276 , 0.04454937],
                           'probePhiWidth':[0.01747818, 0.05881307, 0.10398225, 0.1476358,  0.17033959, 0.21485587, 0.23323278, 0.2530387 , 0.27838778, 0.28815062, 0.29578135, 0.30419102, 0.30908645, 0.30445053, 0.29042994, 0.28126225, 0.25891057, 0.22414481, 0.1701449 , 0.10972719, 0.0296733 ],
                           'probeSigmaIeIe':[0.02933749, 0.08730696, 0.13539114, 0.1825207 , 0.21035164, 0.23582211, 0.25517362, 0.27269423, 0.27799643, 0.28595583, 0.28621528, 0.28923645, 0.28050118, 0.26662262, 0.25331252, 0.245439  , 0.22352915, 0.19660701, 0.16488347, 0.10595295, 0.05362658],
                           'probeEtaWidth':[0.01907875, 0.06893633, 0.11647892, 0.15931963, 0.1903759 , 0.22188218, 0.2480134 , 0.26508688, 0.28200306, 0.29700695, 0.30305111, 0.30395387, 0.30223482, 0.29147233, 0.28783524, 0.2705576 , 0.23825689, 0.2074619, 0.16217982, 0.10611298, 0.0386207 ]})
    return (np.max(qwsIdf[target])/np.array(qwsIdf[target]))#**2

def test(X, Y, qs, qweights, model_from, scale_par):
    Y = np.array(Y)
    predY = predict(X, qs, qweights, model_from)
    es = (Y-predY.T).T
    loss = np.maximum(qs*es, (qs-1.)*es)*qweights
    return np.mean(scale_par['sigma']*np.log(predY)/np.log(300) + scale_par['mu'],axis=0), loss
    

def main(options):
    variables = ['probeS4','probeR9','probeCovarianceIeIp','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 

    data_key = 'data'
    EBEE = 'EB'
      
    inputtrain = 'df_{}_{}_train.h5'.format(data_key, EBEE)
    inputtest = 'df_{}_{}_test.h5'.format(data_key, EBEE)
    
    #load dataframe
    df_train = (pd.read_hdf(inputtrain).loc[:,kinrho+variables])#.sample(100, random_state=100).reset_index(drop=True)
    df_test_raw  = (pd.read_hdf(inputtest).loc[:,kinrho+variables])#.sample(100, random_state=100).reset_index(drop=True)
    
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
    df_train['probeS4'] = np.power(300, df_train['probeS4'])
    df_test_raw['probeS4'] = np.power(300, df_test_raw['probeS4'])

    scale_file = 'scale_para/data_{}_try.h5'.format(EBEE)
    scale_par = gen_scale_par(df_train, kinrho+variables, scale_file)
#    try: 
#        scale_par = pd.read_hdf(scale_file)
#    except FileNotFoundError:
#        scale_par = gen_scale_par(df_train, kinrho+variables, scale_file)
    #scale data

    df_train = scale(df_train, scale_file=scale_file)
    df_test_raw = scale(df_test_raw, scale_file=scale_file)
        
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
#    qweights = get_qweights(target)
#    qweights = 0.25/(qs*np.flip(qs))
    print('quantile loss weights: {}'.format(qweights))

    trainQuantile(X, Y, qs, num_hidden_layers, num_units, act, qweights, dropout, gauss_std, batch_size = 8192, epochs=10, 
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
        Y_test = df_test.loc[:,target]
     
        q_pred, loss_ = test(X_test, Y_test, qs, qweights, model_from='combined_models/{}_{}_{}'.format(data_key, EBEE, target), scale_par=target_scale_par)
        if i==0:
            loss = loss_ 
        else: 
            loss = np.append(loss, loss_, axis=0)


        fig = plt.figure(tight_layout=True)
        plt.hist((target_scale_par['sigma']*np.log(df_test[target])/np.log(300)+target_scale_par['mu']), bins=100, density=True, cumulative=True, histtype='step')
        plt.plot(q_pred, qs, 'o')
        fig.savefig('plots/combined_{}_{}_{}_{}.png'.format(data_key, EBEE, target, str(i)))
#        fig.savefig('plots/combined_{}_{}_{}_{}.pdf'.format(data_key, EBEE, target, str(i)))
        plt.close(fig)
    
    print(loss.shape)
    loss_mean = np.mean(loss, axis=0)
    loss_std = np.std(loss, axis=0)
    print('mean: {}\nstd: {}'.format(loss_mean,loss_std))
    fig2 = plt.figure(tight_layout=True)
    for i in range(len(qs)):
        plt.hist(loss[:,1], bins=100, histtype='step', label='q:{}'.format(qs[i]))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig2.savefig('plots/loss_{}_{}_{}.png'.format(data_key,EBEE,target))
    plt.close(fig2)

 


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-i','--ith_var', action='store', type=int, required=True)
    options = parser.parse_args()
    main(options)
