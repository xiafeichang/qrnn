import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle as pkl
import gzip
from joblib import delayed, Parallel, parallel_backend, register_parallel_backend 

from dataMCCorrectionQRNN import dataMCCorrector
from qrnn import trainQuantile, predict
from mylib.Corrector import Corrector, applyCorrection
from mylib.Shifter import Shifter, applyShift
from mylib.Shifter2D import Shifter2D, apply2DShift
from mylib.tools import *



class dataMCCorrectorIso(dataMCCorrector):

    def trainClfp0t(self, data_key, target, clf_name, n_jobs=1, **kwargs):

        df = self.dfs[data_key]
        features = self.kinrho

        df['p0t_{}'.format(target)] = np.apply_along_axis(lambda x: 0 if x==0 else 1, 0, df[target].values.reshape(1,-1))

        X_train = df.loc[:,features].values
        Y_train = df['p0t_{}'.format(target)].values
        sample_weight_train = df['ml_weight'].values

        clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, maxDepth=10, subsample=0.5, gamma=0, n_jobs=n_jobs, **kwargs)
        clf.fit(X_train, Y_train, sample_weight=sample_weight_train)

        pkl.dump(clf, gzip.open(f'{self.modeldir}/{clf_name}','wb'), protocol=pkl.HIGHEST_PROTOCOL)


    def trainClf3Cat(self, data_key, targets, clf_name, n_jobs=1, **kwargs):

        df = self.dfs[data_key]
        features = self.kinrho

        df['ChIsoCat'] = self.get_class_3Cat(df[targets[0]].values,df[targets[1]].values)

        X_train = df.loc[:,features].values
        Y_train = df['ChIsoCat'].values
        sample_weight_train = df['ml_weight'].values

        clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, maxDepth=10, gamma=0, n_jobs=n_jobs, **kwargs)
        clf.fit(X_train, Y_train, sample_weight=sample_weight_train)

        pkl.dump(clf, gzip.open(f'{self.modeldir}/{clf_name}','wb'), protocol=pkl.HIGHEST_PROTOCOL)


    @staticmethod
    def get_class_3Cat(x,y):
        return [0 if x[i]==0 and y[i]==0 else (1 if x[i]==0 and y[i]>0 else 2) for i in range(len(x))]


    def trainTailRegressor(self, data_key, target, variables, *args, retrain=True, history_fig=None, **kwargs)

        features = self.kinrho + [var for var in variables if var != target]
        df_tail = self.dfs[data_key].query(f'{target}!=0').reset_index(drop=True)

        X = df_tail.loc[:,features]
        Y = df_tail.loc[:,target]
        sample_weight = df_tail.loc[:,'ml_weight']

        qs = self.qs
        qweights = compute_qweights(Y, qs, sample_weight)

        model_file = '{}/{}_{}_tReg_{}'.format(self.modelDir, data_key, self.EBEE, target)
        if os.path.exists(model_file) and not retrain:  
            print(f'model {model_file} already exist, skip training')
        else: 
            print(f'training new {data_key} tail regressor for {target}')
            history, eval_results = trainQuantile(
                X, Y, 
                qs, qweights, 
                self.num_hidden_layers, 
                self.num_units, 
                self.acts, 
                self.num_connected_layers,
                *args, 
                sample_weight = sample_weight,
                epochs = 1000, 
                save_file = model_file, 
                **kwargs, 
                )

            if history_fig is not None: 
                self.drawTrainingHistories(history, history_fig)


    def Shift(self, var, n_jobs=10):
        # Note! the dataframes should be transformed before calling this method. the same for Shift2D below
        print(f'shifting mc with classifiers and tail regressor for {isoVarsPh}')

        X = inverse_transform(self.dfs['mc'].loc[:,self.kinrho], self.transformer, self.kinrho)
        Y = self.dfs['mc'].loc[:,var]

        X_tail = self.dfs['mc'].loc[:,self.kinrho]

        clf_mc = pkl.load(gzip.open('{}/mc_{}_clf_{}.pkl'.format(self.modeldir, self.EBEE, var)))
        clf_data = pkl.load(gzip.open('{}/data_{}_clf_{}.pkl'.format(self.modeldir, self.EBEE, var)))
        tail_regresssor = '{}/mc_{}_{}'.format(self.modeldir, self.EBEE, var)

        self.dfs['mc'][f'{var}_shift'] = parallelize2(applyShift,
            X, Y, X_tail,
            clf_mc, clf_data, tail_regresssor,
            n_jobs = n_jobs, 
            final_reg = False,
            ) 


    def Shift2D(self, variables, n_jobs=10):
        print(f'shifting mc with classifiers and tail regressor for {isoVarsCh}')
        # VERY IMPORTANT! Note the order of targets here

        X = inverse_transform(self.dfs['mc'].loc[:,self.kinrho], self.transformer, self.kinrho)
#        Y = self.dfs['mc'].loc[:,variables] # must be this order: df_mc.loc[:,['probeChIso03','probeChIso03worst']]
        Y = self.dfs['mc'].loc[:,['probeChIso03','probeChIso03worst']]

        X_tail = self.dfs['mc'].loc[:,self.kinrho]

        clf_mc = pkl.load(gzip.open('{}/mc_{}_clf_{}_{}.pkl'.format(self.modeldir, self.EBEE, variables[0], variables[1])))
        clf_data = pkl.load(gzip.open('{}/data_{}_clf_{}_{}.pkl'.format(self.modeldir, self.EBEE, variables[0], variables[1])))

        Y_shifted = parallelize2(apply2DShift,
            X, Y, X_tail,
            clf_mc, clf_data, 
            '{}/mc_{}_tReg_probeChIso03'.format(modeldir, EBEE), 
            '{}/mc_{}_tReg_probeChIso03worst'.format(modeldir, EBEE), 
            n_jobs = n_jobs, 
            final_reg = False,
            ) 
        self.dfs['mc']['probeChIso03_shift'] = Y_shifted[:,0]
        self.dfs['mc']['probeChIso03worst_shift'] = Y_shifted[:,1]


    def Correct(self, target, n_jobs=10): 

        if isinstance(target, str) or len(target)==1: 
            self.Shift(target, n_jobs=n_jobs)
            super(dataMCCorrector, self).Correct(target, n_jobs=n_jobs)
        elif isinstance(target, list) and len(target)==2: 
            self.Shift2D(target, n_jobs=n_jobs)
            for var in target: 
                super(dataMCCorrector, self).Correct(var, n_jobs=n_jobs)
        else: 
            raise(f'Argument "target" must be a list with length 1 or 2, or a str, recieved: {target}')


    def train(self,):  


    def trainFinal(self,): 


    def CorrectFinal(self, ):

