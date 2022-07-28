import warnings 
import argparse
import os
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('cairo')
from matplotlib import pyplot as plt
from sklearn import preprocessing 
import pickle
import gzip

from qrnn import trainQuantile, predict, scale
from clf_Iso import trainClfp0t, trainClf3Cat
from mylib.transformer import fit_standard_scaler, fit_quantile_transformer, fit_power_transformer, transform, inverse_transform
from mylib.IdMVAComputer import helpComputeIdMva
from mylib.tools import *



class quantileRegressionNeuralNet(object): 
    def __init__(self, variables, EBEE, modelDir, plotsDir,quantiles=None, weight_name=None):
        self.kinrho = ['probePt','probeScEta','probePhi','rho']
        self.variables = variables
        self.weight_name = weight_name

        self.EBEE = EBEE

        self.modelDir = modelDir
        self.plotsDir = plotsDir

        self.dfs = {}

        if quantiles is None: 
            self.qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
        else: 
            self.qs = quantiles

    def loadDatasets(self, datasetDir, dataName, mcName, nEvt=None, random_state=100):
        if nEvt is None: 
            self.dfs['data'] = pd.read_hdf(f'{datasetDir}/{dataName}').reset_index(drop=True) 
            self.dfs['mc'] = pd.read_hdf(f'{datasetDir}/{mcName}').reset_index(drop=True) 
        else: 
            self.dfs['data'] = pd.read_hdf(f'{datasetDir}/{dataName}').sample(nEvt, random_state=random_state).reset_index(drop=True) 
            self.dfs['mc'] = pd.read_hdf(f'{datasetDir}/{mcName}').sample(nEvt, random_state=random_state).reset_index(drop=True) 

    def transform(self, trans_vars, transformer, trans_type='standard', **kwargs): 
        self.trans_vars = trans_vars
        self.transformer = transformer
        try: 
            self.dfs['data'].loc[:, trans_vars] = transform(self.dfs['data'].loc[:, trans_vars], transformer, trans_vars)
            self.dfs['mc'].loc[:, trans_vars] = transform(self.dfs['mc'].loc[:, trans_vars], transformer, trans_vars)
        except FileNotFoundError: 
            if trans_type == 'quantile':
                fit_quantile_transformer(self.dfs['data'].loc[:, trans_vars], trans_vars, transformer, **kwargs)
            elif trans_type == 'power'
                fit_power_transformer(self.dfs['data'].loc[:, trans_vars], trans_vars, transformer, **kwargs)
            else: 
                warnings.warn(f'Wrong argument "trans_type", argument must be in ["standard", "quantile", "power"]. Now using defalt argument "standard"')
                fit_standard_scaler(self.dfs['data'].loc[:, trans_vars], trans_vars, transformer)

            self.dfs['data'].loc[:, trans_vars] = transform(self.dfs['data'].loc[:, trans_vars], transformer, trans_vars)
            self.dfs['mc'].loc[:, trans_vars] = transform(self.dfs['mc'].loc[:, trans_vars], transformer, trans_vars)

    def set_transformer(self, transformer):
        self.transformer = transformer

    def get_transformer(self):
        return self.transformer

    def setupNN(self, num_hidden_layers, num_connected_layers, num_units, activations, dropout=None, gauss_std=None):
        self.num_hidden_layers = num_hidden_layers
        self.num_connected_layers = num_connected_layers
        self.num_units = num_units
        self.acts = activations
        self.dropout = dropout
        self.gauss_std = gauss_std

    def train(self, data_key, *args, targets=None, **kwargs, retrain=True, history_fig=None): 

        if targets is None: 
            targets = self.variables

        if isinstance(targets, str): 
            self._train1var(data_key, *args, targets, self.variables, **kwargs, retrain=retrain, history_fig=history_fig)
        else: 
            for target in targets: 
                self._train1var(data_key, *args, target, targets, **kwargs, retrain=retrain, history_fig=history_fig)

    def _train1var(self, data_key, *args, target, variables, **kwargs, retrain=True, history_fig=None)

        if data_key == 'mc':
            features = self.kinrho + [f'{x}_corr' for x in variables[:variables.index(target)]]
        else: 
            features = self.kinrho + variables[:variables.index(target)]

        X = self.dfs[data_key].loc[:,features]
        Y = self.dfs[data_key].loc[:,target]
        sample_weight = self.dfs[data_key].loc[:,'ml_weight']

        qs = self.qs
        qweights = compute_qweights(Y, qs, sample_weight)

        model_file = '{}/{}_{}_{}'.format(self.modelDir, data_key, self.EBEE, target)
        if os.path.exists(model_file) and not retrain:  
            print(f'model {model_file} already exist, skip training')
        else: 
            print(f'training new mc model for {target}')
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

        if data_key == 'mc':
            self.Correct(target, features)


    def Correct(self, target, features=None, df=None, n_jobs=10): 

        diz = 'Iso' in target or 'esEnergy' in target

        models_mc = '{}/{}_{}_{}'.format(self.modelDir, 'mc', self.EBEE, target)
        models_d = '{}/{}_{}_{}'.format(self.modelDir, 'data', self.EBEE, target)

        if features is None: 
            features = kinrho + [f'{x}_corr' for x in self.variables[:self.variables.index(target)]]

        print(f'Correct {target} using features {features} with models： {models_d}, {models_mc} and diz={diz}')
        if df is None: 
            X = self.dfs['mc'].loc[:,features]
            Y = self.dfs['mc'].loc[:,target]
            self.dfs['mc'][f'{target}_corr'] = parallelize(applyCorrection, X, Y, models_mc, models_d, diz=diz, n_jobs=10)
        else: 
            X = df.loc[:,features]
            Y = df.loc[:,target]
            df[f'{target}_corr'] = parallelize(applyCorrection, X, Y, models_mc, models_d, diz=diz, n_jobs=10)


    def CorrectDatasets(self, outputDir, outputName, inputDir=None, inputName=None, trans_vars=None, transformer=None, n_jobs=10): 

        if inputName is not None: 

            df = pd.read_hdf(f'{inputDir}/{inputName}')#.reset_index(drop=True)

            if trans_vars is None: 
                trans_vars = self.trans_vars
            if transformer is None: 
                transformer = self.transformer
            df.loc[:, trans_vars] = transform(df.loc[:, trans_vars], transformer, trans_vars)

            for target in variables: 
                self.Correct(target, df=df, n_jobs=n_jobs)

            df.loc[:,trans_vars] = inverse_transform(df.loc[:,trans_vars], transformer, trans_vars)
            df.to_hdf(f'{outputDir}/{outputName}','df',mode='w',format='t')
        else: 
            for target in variables: 
                self.Correct(target, n_jobs=n_jobs)

            self.dfs['mc'].loc[:,self.trans_vars] = inverse_transform(self.dfs['mc'].loc[:,self.trans_vars], self.transformer, self.trans_vars)
            self.dfs['mc'].to_hdf(f'{outputDir}/{outputName}','df',mode='w',format='t')



    def trainFinal(self, *args, **kwargs): 
        pass

    @staticmethod
    def standardScale(df, fileName, varNames):
        try: 
            df.loc[:,varNames] = transform(df.loc[:,varNames], fileName, varNames)
        except FileNotFoundError: 
            fit_standard_scaler(df.loc[:,varNames], varNames, fileName)
            df.loc[:,varNames] = transform(df.loc[:,varNames], fileName, varNames)
        return df

    @staticmethod
    def drawTrainingHistories(history, figname):
        fig = plt.figure(tight_layout=True)
        plt.plot(history.history['loss'], label='training')
        plt.plot(history.history['val_loss'], label='validation')
        plt.yscale('log')
        plt.title('Training history')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        fig.savefig(f'{figname}.pdf')
        fig.savefig(f'{figname}.png')



