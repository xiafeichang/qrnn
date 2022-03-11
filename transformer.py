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
        ax.hist(df[var], bins=100)
        ax.ticklabel_format(style='sci', scilimits=(-2,3), axis='both')
        ax.set_title(var)
    fig.savefig('plots/transform_{}.png'.format(file_name))

def fit_transformer(df, variables, file_name, output_distribution='uniform', random_state=100):
    df_tr = pd.DataFrame()
    for var in variables: 
        transformer = preprocessing.QuantileTransformer(output_distribution=output_distribution, random_state=random_state)
        df_tr[var] = transformer.fit_transform(np.array(df[var]).reshape(-1,1)).flatten()
        pickle.dump(transformer, gzip.open('transformer/{}_{}.pkl'.format(file_name, var), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    return df_tr

def transform(df, file_name, variables):
    if len(df.shape)==1 or df.shape[1]==1:
        transformer = pickle.load(gzip.open('transformer/{}_{}.pkl'.format(file_name, variables)))
        return transformer.transform(np.array(df).reshape(-1,1)).flatten()
    elif isinstance(variables, str): 
        transformer = pickle.load(gzip.open('transformer/{}_{}.pkl'.format(file_name, variables)))
        return transformer.transform(np.array(df).reshape(-1,1)).reshape(df.shape)
    else: 
        df_tr = pd.DataFrame()
        for var in variables: 
            transformer = pickle.load(gzip.open('transformer/{}_{}.pkl'.format(file_name, var)))
            df_tr[var] = transformer.transform(np.array(df[var]).reshape(-1,1)).flatten()
        return df_tr

def inverse_transform(df, file_name, variables):
    if len(df.shape)==1 or df.shape[1]==1:
        transformer = pickle.load(gzip.open('transformer/{}_{}.pkl'.format(file_name, variables)))
        return transformer.inverse_transform(np.array(df).reshape(-1,1)).flatten()
    elif isinstance(variables, str): 
        transformer = pickle.load(gzip.open('transformer/{}_{}.pkl'.format(file_name, variables)))
        return transformer.inverse_transform(np.array(df).reshape(-1,1)).reshape(df.shape)
    else: 
        df_itr = pd.DataFrame()
        for var in variables: 
            transformer = pickle.load(gzip.open('transformer/{}_{}.pkl'.format(file_name, var)))
            df_itr[var] = transformer.inverse_transform(np.array(df[var]).reshape(-1,1)).flatten()
        return df_itr

