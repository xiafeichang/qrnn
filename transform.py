import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn import preprocessing 
import pickle
import gzip
import time

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
    try: 
        sample_weight = np.array(df['ml_weight'])
    except KeyError:
        print('No weight found!')
        sample_weight = None
    for var in variables: 
        transformer = preprocessing.StandardScaler()
        transformer.fit(np.array(df[var]).reshape(-1,1), sample_weight=sample_weight)
        pickle.dump(transformer, gzip.open('transformer/{}_{}.pkl'.format(file_name, var), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

def fit_quantile_transformer(df, variables, file_name, output_distribution='uniform', random_state=None):
    for var in variables: 
        transformer = preprocessing.QuantileTransformer(output_distribution=output_distribution, random_state=random_state)
        if 'Iso' in var: 
            print(f'transforming isolation variable {var}, ignoring {var}==0')
            transformer.fit(np.array(df.loc[df[var]!=0,var]).reshape(-1,1)) # for isolation variables
        else: 
            transformer.fit(np.array(df.loc[:,var]).reshape(-1,1))
        pickle.dump(transformer, gzip.open('transformer/{}_{}.pkl'.format(file_name, var), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

def fit_power_transformer(df, variables, file_name, methods = None):
    if methods is None: 
        methods = ['yeo-johnson' for _ in variables]
    for var, method in zip(variables, methods): 
        transformer = preprocessing.PowerTransformer(method=method, standardize=True)
        if 'Iso' in var: 
            print(f'transforming isolation variable {var}, ignoring {var}==0')
            transformer.fit(np.array(df.loc[df[var]!=0,var]).reshape(-1,1)) # for isolation variables
        else: 
            transformer.fit(np.array(df.loc[:,var]).reshape(-1,1))
        pickle.dump(transformer, gzip.open('transformer/{}_{}.pkl'.format(file_name, var), 'wb'),protocol=pickle.HIGHEST_PROTOCOL)

def transform(df, file_name, variables):
    if len(df.shape)==1 or df.shape[1]==1:
        var_raw = variables[:variables.find('_')] if '_' in variables else variables
        transformer = pickle.load(gzip.open('transformer/{}_{}.pkl'.format(file_name, var_raw)))
        return transformer.transform(np.array(df).reshape(-1,1)).flatten()
    else: 
        df_tr = pd.DataFrame()
        for var in variables: 
            var_raw = var[:var.find('_')] if '_' in var else var
            transformer = pickle.load(gzip.open('transformer/{}_{}.pkl'.format(file_name, var_raw)))
            df_tr[var] = transformer.transform(np.array(df[var]).reshape(-1,1)).flatten()
        try: 
            df_tr['ml_weight'] = df['ml_weight']
        except KeyError:
            print('No weight found! ')
        return df_tr

def inverse_transform(df, file_name, variables):
    if len(df.shape)==1 or df.shape[1]==1:
        var_raw = variables[:variables.find('_')] if '_' in variables else variables
        transformer = pickle.load(gzip.open('transformer/{}_{}.pkl'.format(file_name, variables)))
        return transformer.inverse_transform(np.array(df).reshape(-1,1)).flatten()
    else: 
        df_itr = pd.DataFrame()
        for var in variables: 
            var_raw = var[:var.find('_')] if '_' in var else var
            transformer = pickle.load(gzip.open('transformer/{}_{}.pkl'.format(file_name, var_raw)))
            df_itr[var] = transformer.inverse_transform(np.array(df[var]).reshape(-1,1)).flatten()
        try: 
            df_itr['ml_weight'] = df['ml_weight']
        except KeyError:
            print('No weight found! ')
        return df_itr


def main():
#    variables = ['probeS4','probeR9','probeCovarianceIeIp','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
#    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    weight = ['ml_weight']
#    
#    vars_tran = kinrho + variables

    vars_tran = ['probePhoIso','probeChIso03','probeChIso03worst']
    
    data_key = 'data'
    EBEE = 'EB'
      
    inputtrain = 'weighted_dfs/df_{}_{}_train.h5'.format(data_key, EBEE)
    inputtest = 'weighted_dfs/df_{}_{}_test.h5'.format(data_key, EBEE)
    
    #load dataframe
    nEvnt = 5000000
    df_train = (pd.read_hdf(inputtrain).loc[:,vars_tran+weight]).sample(nEvnt, random_state=100).reset_index(drop=True)
    df_test  = (pd.read_hdf(inputtest).loc[:,vars_tran+weight]).sample(nEvnt, random_state=100).reset_index(drop=True)
    print(df_train)
    print(df_test)
    
    # show the distribution before transform
    matplotlib.use('agg')
    nrows = 1
    ncols = 3
    figsize = (12,3)
    showDist(df_train, vars_tran, 'Original distribution (training set)', 'train_before', nrows=nrows, ncols=ncols, figsize=figsize)
    showDist(df_test, vars_tran, 'Original distribution (test set)', 'test_before', nrows=nrows, ncols=ncols, figsize=figsize)
    
    # transform
    #methods = ['box-cox','box-cox','yeo-johnson','box-cox','yeo-johnson','box-cox','box-cox']
    transformer_file = '{}_{}'.format(data_key, EBEE)
    fit_start = time.time()
    fit_quantile_transformer(df_train, vars_tran, transformer_file, output_distribution='normal', random_state=100)
    #fit_power_transformer(df_train, vars_tran, transformer_file, methods)
    #fit_standard_scaler(df_train, kinrho, transformer_file)
    fit_end = time.time()
    print('time spent in fitting the transformer: {} s'.format(fit_end-fit_start))
    
    # draw transformed distributions
    df_train_tr = transform(df_train, transformer_file, vars_tran)
    df_test_tr = transform(df_test, transformer_file, vars_tran)
    print(df_train_tr)
    print(df_test_tr)
    trans_end = time.time()
    print('time spent in transforming the test dataset: {} s'.format(trans_end-fit_end))
    
    showDist(df_train_tr, vars_tran, 'Transformed distribution (training set)', 'train_after', nrows=nrows, ncols=ncols, figsize=figsize)
    showDist(df_test_tr, vars_tran, 'Transformed distribution (test set)', 'test_after', nrows=nrows, ncols=ncols, figsize=figsize)
    
    # inverse transform
    df_train_itr = inverse_transform(df_train_tr, transformer_file, vars_tran)
    df_test_itr = inverse_transform(df_test_tr, transformer_file, vars_tran)
    print(df_train_itr)
    print(df_test_itr)
    
    showDist(df_train_itr, vars_tran, 'Inversed transform (training set)', 'train_inverse', nrows=nrows, ncols=ncols, figsize=figsize)
    showDist(df_test_itr, vars_tran, 'Inversed transform (test set)', 'test_inverse', nrows=nrows, ncols=ncols, figsize=figsize)
    
    #showDist((df_train-df_train_itr)/df_train.std(), vars_tran, 'Original - inversed transform (training set)', 'train_diff', nrows=3, ncols=4, figsize=(12,9))
    #showDist((df_test-df_test_itr)/df_test.std(), vars_tran, 'Original - inversed transform (test set)', 'test_diff', nrows=3, ncols=4, figsize=(12,9))




if __name__ == "__main__": 
    main()
 
