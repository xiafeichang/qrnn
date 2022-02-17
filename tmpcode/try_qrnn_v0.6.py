import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from qrnn import trainQuantile, predict


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

#train for quantile q
#qs = [0.1, 0.5, 0.9]
#num_hidden_layers = 2
#num_units = [10, 10]
#act = ['tanh', 'tanh']

qs = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
num_hidden_layers = 5
num_units = [500, 300, 200, 100, 50]
act = ['tanh','exponential', 'softplus', 'tanh', 'elu']

scale_par = 'scale_para/{}.h5'.format(inputfile[3:-12])
print('>>>>>>>>> train for variable {}'.format(target))
X = df_train.loc[:,features]
Y = df_train.loc[:,target]
 
for q in qs:
    model_file = 'models/{}_{}'.format(target,str(q).replace('.','p'))
    
    trainQuantile(X,Y,q,num_hidden_layers,num_units,act,batch_size=8192, scale_par=scale_par, save_file=model_file)
    
# test
pTs = [25., 30., 35., 40., 45., 50., 60., 150.]
for i in range(len(pTs)-1): 
    df_test = df_test_raw.query('probePt>' + str(pTs[i]) + ' and probePt<' + str(pTs[i+1]))
    X_test = df_test.loc[:,features]
    
    q_pred = []
    for q in qs:
        model_file = 'models/{}_{}'.format(target,str(q).replace('.','p'))
    
        q_pred.append(np.mean(predict(X_test, model_from=model_file, scale_par=scale_par,target=target)))
    
    #plot the result
    
    fig = plt.figure(tight_layout=True)
    plt.hist(df_test[target], bins=100, density=True, cumulative=True, histtype='step')
    plt.plot(q_pred, qs, 'o')
    fig.savefig('plots/' + target + '_' + str(i) + '.png')
    fig.savefig('plots/' + target + '_' + str(i) + '.pdf')

