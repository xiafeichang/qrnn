import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn import preprocessing 
import pickle
import gzip

def showDist(df, variables, title, file_name, nrows, ncols): 
    fig, axs = plt.subplots(3,4,figsize=(12,6),tight_layout=True)
    fig.suptitle(title)
    for ax, var in zip(axs.flat[:len(variables)-nrows*ncols], variables): 
        ax.hist(df[var], bins=100)
        ax.ticklabel_format(style='sci', scilimits=(-2,3), axis='both')
        ax.set_title(var)
#    plt.show()
    fig.savefig('plots/transform_{}.png'.format(file_name))


variables = ['probeS4','probeR9','probeCovarianceIeIp','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
kinrho = ['probePt','probeScEta','probePhi','rho'] 

data_key = 'data'
EBEE = 'EB'
  
inputtrain = 'df_{}_{}_train.h5'.format(data_key, EBEE)
inputtest = 'df_{}_{}_test.h5'.format(data_key, EBEE)

#load dataframe
df_train = (pd.read_hdf(inputtrain).loc[:,kinrho+variables])#.sample(1000, random_state=100).reset_index(drop=True)
df_test  = (pd.read_hdf(inputtest).loc[:,kinrho+variables])#.sample(1000, random_state=100).reset_index(drop=True)

# show the distribution before transform
matplotlib.use('agg')
showDist(df_train, kinrho+variables, 'Original distribution (training set)', 'before', 3, 4)

# transform
transformer_file = 'transformer/{}_{}.pkl'.format(data_key, EBEE)
transformer = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
df_train_tr = transformer.fit_transform(np.array(df_train).T)
pickle.dump(transformer, gzip.open(transformer_file, 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

# draw transformed distributions
df_train_tr = pd.DataFrame(df_train_tr.T, columns=kinrho+variables)
showDist(df_train_tr, kinrho+variables, 'Transformed distribution (training set)', 'after', 3, 4)


 
