import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from qrnn import QRNN

def main(options):
    
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
      
    inputfile = 'df_data_EB_train.h5'
    n_evt = 2000000
    
    #load dataframe
    df_total = pd.read_hdf(inputfile)
    df_smp = df_total.sample(n_evt, random_state=100).reset_index(drop=True)
    
    df_test_raw  = df_smp[1000000:] 
    
    #set features and target
    features = kinrho 
    target = variables[0]
    
    qs = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    
    scale_para_file = 'scale_para/{}_{}.h5'.format(inputfile[3:-12], target)
    
    # test
    pTlow = options.pT_low
    pThigh = options.pT_high

    df_test = df_test_raw.query('probePt>' + pTlow + ' and probePt<' + pThigh)
    
    q_pred = []
    for q in qs:
        model_file = 'models/{}_{}'.format(target,str(q).replace('.','p'))
    
        qrnn_test = QRNN(df_test, features, target, scale_file=scale_para_file)
        q_pred.append(np.mean(qrnn_test.predict(q, model_from=model_file)))
    
    #plot the result
    
    fig = plt.figure(tight_layout=True)
    plt.hist(df_test[target], bins=100, density=True, cumulative=True, histtype='step')
    plt.plot(q_pred, qs, 'o')
    fig.savefig('plots/' + target + '_pT(' + pTlow + ',' + pThigh + ').png')
    fig.savefig('plots/' + target + '_pT(' + pTlow + ',' + pThigh + ').pdf')

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-l','--pT_low', action='store', type=str, required=True)
    requiredArgs.add_argument('-h','--pT_high', action='store', type=str, required=True)


