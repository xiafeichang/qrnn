import warnings 
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('cairo')
from matplotlib import pyplot as plt
from sklearn import preprocessing 
import pickle
import gzip

from qrnn import trainQuantile, predict
from mylib.transformer import fit_standard_scaler, fit_quantile_transformer, fit_power_transformer, transform, inverse_transform
from mylib.IdMVAComputer import helpComputeIdMva
from mylib.tools import *



class dataMCCorrector(object): 
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

    def cutDatasets(self, cut, data_key='both'): 
        if data_key == 'data' or data_key == 'both': 
            self.dfs['data'] = self.dfs['data'].query(cut)
        elif data_key == 'mc' or data_key == 'both': 
            self.dfs['mc'] = self.dfs['mc'].query(cut)

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
        try: 
            return self.transformer
        except ValueError as e:
            print(e)
            return None

    def setupNN(self, num_hidden_layers, num_units, activations, num_connected_layers=None, dropout=None, gauss_std=None):
        self.num_hidden_layers = num_hidden_layers
        self.num_connected_layers = num_connected_layers
        self.num_units = num_units
        self.acts = activations
        self.dropout = dropout
        self.gauss_std = gauss_std

    def train(self, data_key, *args, targets=None, cut=None, retrain=True, history_fig=None, **kwargs): 

        if targets is None: 
            targets = self.variables

        if isinstance(targets, str): 
            self._train1var(data_key, targets, self.variables, *args, cut=cut, retrain=retrain, history_fig=history_fig, **kwargs)
        else: 
            for target in targets: 
                self._train1var(data_key, target, targets, *args, cut=cut, retrain=retrain, history_fig=history_fig, **kwargs)

    def _train1var(self, data_key, target, variables, *args, cut=None, retrain=False, history_fig=None, **kwargs)

        if cut is None: 
            df = self.dfs[data_key]
        else: 
            df = self.dfs[data_key].query(cut).reset_index(drop=True)

        if data_key == 'mc':
            features = self.kinrho + [f'{x}_corr' for x in variables[:variables.index(target)]]
        else: 
            features = self.kinrho + variables[:variables.index(target)]

        X = df.loc[:,features]
        Y = df.loc[:,target]
        sample_weight = df.loc[:,'ml_weight']

        qs = self.qs
        qweights = compute_qweights(Y, qs, sample_weight)

        model_file = '{}/{}_{}_{}'.format(self.modelDir, data_key, self.EBEE, target)
        if os.path.exists(model_file) and not retrain:  
            print(f'model {model_file} already exist, skip training')
        else: 
            print(f'training new {data_key} model for {target}')
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
            df[f'{target}_corr'] = parallelize(applyCorrection, X, Y, models_mc, models_d, n_jobs=n_jobs, diz=diz)


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



    def trainFinal(self, transformer_features, transformer_targets, var_type, *args, variables=None, train_df=None, cut=None, retrain=False, history_fig=None, **kwargs): 

        if train_df is None: 
            if cut is None:
                self.df_corr = self.dfs['mc']
            else: 
                self.df_corr = self.dfs['mc'].query(cut).reset_index(drop=True)
        else: 
            if cut is None:
                self.df_corr = pd.read_hdf(train_df).reset_index(drop=True) 
            else: 
                self.df_corr = (pd.read_hdf(train_df).query(cut)).reset_index(drop=True)

        if not all(var in self.df_corr.columns for var in [f'{x}_corr' for x in self.variables]):
            raise('The loaded MC dataframe is not completely corrected. Please correct it firstly')

        # compute var_corr - var as target
        if variables is None: 
            variables = self.variables 
        elif not isinstance(variables, list):
            variables = [variables]
        features = self.kinrho+variables

        vars_corr_diff = [f'{var}_corr_diff' for var in variables]
        for var in variables
            self.df_corr[f'{var}_corr_diff'] = [self.df_corr[f'{var}_corr'] - self.df_corr[var]

        self.transformFinal(vars_corr_diff, transformer_targets, features, transformer_features)

        sample_weight = self.df_corr.loc[:,'ml_weight']
        X = self.df_corr.loc[:,features]
        Y = self.df_corr.loc[:,vars_corr_diff]
    
        model_file = '{}/mc_{}_{}_final'.format(self.modeldir, self.EBEE, var_type)
        if os.path.exists(model_file) and not retrain:  
            print(f'model {model_file} already exist, skip training')
        else: 
            print(f'training new final regressor for {target}')
            history, eval_results = trainNN(
                X, Y, 
                self.num_hidden_layers, 
                self.num_units, 
                self.act,
                *args,  
                sample_weight = sample_weight,
                epochs = 1000, 
                save_file = model_file, 
                **kwargs,
                )

            if history_fig is not None: 
                self.drawTrainingHistories(history, history_fig)


    def transformFinal(self, targets, transformer_targets, features=None, transformer_features=None):

        if features is not None and transformer_features is not None: 
            self.df_corr.loc[:,features] = transform(self.df_corr.loc[:,features], transformer_features, features)
    
        try: 
            self.df_corr.loc[:,targets] = transform(self.df_corr.loc[:,targets], transformer_targets, targets)
        except FileNotFoundError: 
            fit_standard_scaler(self.df_corr.loc[:,targets], targets, transformer_targets)
            self.df_corr.loc[:,targets] = transform(self.df_corr.loc[:,targets], transformer_targets, targets)


    def CorrectFinal(self, outputDir, outputName, variables, transformer_diff, inputDir=None, inputName=None, transformer_features=None, transformed=False): 

        if variables is None: 
            variables = self.variables 
        elif not isinstance(variables, list):
            variables = [variables]
        features = self.kinrho+variables

        vars_corr_diff = [f'{var}_corr_diff' for var in variables]


        if inputName is not None: 
            df = self.dfs['mc']
            if transformer_features is not None and not transformed: 
                df.loc[:, features] = transform(df.loc[:, features], transformer_features, features)

        else:
            df = pd.read_hdf(f'{inputDir}/{inputName}')#.reset_index(drop=True)
        if transformer_features is not None: 
            df.loc[:, features] = transform(df.loc[:, features], transformer_features, features)

        model_file = '{}/mc_{}_{}_final'.format(self.modeldir, self.EBEE, var_type)
        df_diff = predict(df.loc[:, features], model_file, transformer_diff, vars_corr_diff) 

        if transformer_features is not None: 
            df.loc[:,features] = inverse_transform(df.loc[:,features], transformer_features, features)

        for var in variables:
            df[f'{var}_corr_final'] = df[var] + df_diff[f'{var}_corr_diff']

        df.to_hdf(f'{outputDir}/{outputName}','df',mode='w',format='t')

 
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



