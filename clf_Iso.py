import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from time import time
import pickle as pkl
import gzip
from joblib import delayed, Parallel, parallel_backend, register_parallel_backend 


def trainClfp0t(df, features, target, clf_name, val_split=0.1, tree_method='auto', n_jobs=1, backend='loky', **kwargs):

    df['p0t_{}'.format(target)] = np.apply_along_axis(lambda x: 0 if x==0 else 1, 0, df[target].values.reshape(1,-1))
#    df_train, df_val = train_test_split(df, test_size=val_split, random_state=100)

    X_train = df.loc[:,features].values
    Y_train = df['p0t_{}'.format(target)].values
    sample_weight_train = df['ml_weight'].values
#    X_train = df_train.loc[:,features]
#    Y_train = df_train.loc[:,'p0t_{}'.format(target)]
#    sample_weight_train = np.sqrt(df_train.loc[:,'ml_weight'])

#    X_val = df_val.loc[:,features]
#    Y_val = df_val.loc[:,'p0t_{}'.format(target)]
#    sample_weight_val = np.sqrt(df_val.loc[:,'ml_weight'])

    clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, maxDepth=10, subsample=0.5, gamma=0, tree_method=tree_method, n_jobs=n_jobs)    # max_depth=10, tree_method=tree_method
#    with parallel_backend(backend):
    clf.fit(
        X_train, Y_train, 
        sample_weight=sample_weight_train,
#        eval_set = [(X_train,Y_train), (X_val, Y_val)],
#        sample_weight_eval_set = [sample_weight_train, sample_weight_val],
#        **kwargs,
        )

    dic = {'clf': clf, 'X': features, 'Y': target}
    pkl.dump(dic,gzip.open(clf_name,'wb'),protocol=pkl.HIGHEST_PROTOCOL)

#    return clf.evals_result()
    return 0 

def trainClf3Cat(df, features, target, clf_name, val_split=0.1, tree_method='auto', backend='loky', n_jobs=1, **kwargs):

    df['ChIsoCat'] = get_class_3Cat(df[target[0]].values,df[target[1]].values)
#    df_train, df_val = train_test_split(df, test_size=val_split, random_state=100)

    X_train = df.loc[:,features].values
    Y_train = df['ChIsoCat'].values
    sample_weight_train = df['ml_weight'].values
#    X_train = df_train.loc[:,features]
#    Y_train = df_train.loc[:,'ChIsoCat']
#    sample_weight_train = np.sqrt(df_train.loc[:,'ml_weight'])

#    X_val = df_val.loc[:,features]
#    Y_val = df_val.loc[:,'ChIsoCat']
#    sample_weight_val = np.sqrt(df_val.loc[:,'ml_weight'])

    clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, maxDepth=10, gamma=0, tree_method=tree_method, n_jobs=n_jobs)    # max_depth=10, tree_method=tree_method
#    with parallel_backend(backend):
    clf.fit(
        X_train, Y_train, 
        sample_weight=sample_weight_train,
#        eval_set = [(X_train,Y_train), (X_val, Y_val)],
#        sample_weight_eval_set = [sample_weight_train, sample_weight_val],
#        **kwargs,
        )

    dic = {'clf': clf, 'X': features, 'Y': target}
    pkl.dump(dic,gzip.open(clf_name,'wb'),protocol=pkl.HIGHEST_PROTOCOL)

#    return clf.evals_result()
    return 0 

def get_class_3Cat(x,y):
    return [0 if x[i]==0 and y[i]==0 else (1 if x[i]==0 and y[i]>0 else 2) for i in range(len(x))]


