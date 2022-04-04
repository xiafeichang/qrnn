import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn import preprocessing 
import pickle
import gzip

def showDist(df, variables, title, file_name, nrows, ncols, figsize): 
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,tight_layout=True)
    fig.suptitle(title)
    for ax, var in zip(axs.flat, variables): 
        ax.hist(df[var], bins=100, weights=df['ml_weight'])
        ax.ticklabel_format(style='sci', scilimits=(-2,3), axis='both')
        ax.set_title(var)
#        ax.annotate('mean: {}\nstd:{}'.format(df[var].mean(), df[var].std()), (0.2,0.8), xycoords='axes fraction')
#    plt.show()
    fig.savefig('plots/transform_{}.png'.format(file_name))

def fit_standard_scaler(df, variables, file_name):
    for var in variables: 
        transformer = preprocessing.StandardScaler()
        transformer.fit(np.array(df[var]).reshape(-1,1), sample_weight=np.array(df['ml_weight']))
        pickle.dump(transformer, gzip.open('transformer/{}_{}.pkl'.format(file_name, var), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

def fit_quantile_transformer(df, variables, file_name, output_distribution='uniform', random_state=None):
    for var in variables: 
        transformer = preprocessing.QuantileTransformer(output_distribution=output_distribution, random_state=random_state)
        transformer.fit(np.array(df[var]).reshape(-1,1))
        pickle.dump(transformer, gzip.open('transformer/{}_{}.pkl'.format(file_name, var), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

def fit_power_transformer(df, variables, file_name, methods = None):
    if methods is None: 
        methods = ['yeo-johnson' for _ in variables]
    for var, method in zip(variables, methods): 
        transformer = preprocessing.PowerTransformer(method=method, standardize=True)
        transformer.fit(np.array(df[var]).reshape(-1,1))
        pickle.dump(transformer, gzip.open('transformer/{}_{}.pkl'.format(file_name, var), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

def transform(df, file_name, variables):
    if len(df.shape)==1 or df.shape[1]==1:
        transformer = pickle.load(gzip.open('transformer/{}_{}.pkl'.format(file_name, variables)))
        return transformer.transform(np.array(df).reshape(-1,1)).flatten()
    else: 
        df_tr = pd.DataFrame()
        for var in variables: 
            transformer = pickle.load(gzip.open('transformer/{}_{}.pkl'.format(file_name, var)))
            df_tr[var] = transformer.transform(np.array(df[var]).reshape(-1,1)).flatten()
        df_tr['ml_weight'] = df['ml_weight']
        return df_tr

def inverse_transform(df, file_name, variables):
    if len(df.shape)==1 or df.shape[1]==1:
        transformer = pickle.load(gzip.open('transformer/{}_{}.pkl'.format(file_name, variables)))
        return transformer.inverse_transform(np.array(df).reshape(-1,1)).flatten()
    else: 
        df_itr = pd.DataFrame()
        for var in variables: 
            transformer = pickle.load(gzip.open('transformer/{}_{}.pkl'.format(file_name, var)))
            df_itr[var] = transformer.inverse_transform(np.array(df[var]).reshape(-1,1)).flatten()
        df_itr['ml_weight'] = df['ml_weight']
        return df_itr

