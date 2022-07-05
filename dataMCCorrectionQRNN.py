import argparse
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
from mylib.transformer import transform, inverse_transform
from mylib.IdMVAComputer import helpComputeIdMva
from mylib.tools import *



class quantileRegressionNeuralNet(object): 
    def __init__(self, variables, EBEE, tranformerDir, modelDir, plotsDir, outDir, quantiles=None, weight_name=None):
        self.kinrho = ['probePt','probeScEta','probePhi','rho']
        self.variables = variables
        self.weight_name = weight_name

        self.modelDir = modelDir
        self.outDir = outDir

        if quantiles is None: 
            self.qs = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99])
        else: 
            self.qs = quantiles

    def loadTrainingData(self, datasetDir, dataName, mcName, nEvt=None, random_state=100):
        if nEvt is None: 
            self.data_train = pd.read_hdf(f'{datasetDir}/{dataName}').reset_index(drop=True) 
            self.mc_train = pd.read_hdf(f'{datasetDir}/{mcName}').reset_index(drop=True) 
        else: 
            self.data_train = pd.read_hdf(f'{datasetDir}/{dataName}').sample(nEvt, random_state=random_state).reset_index(drop=True) 
            self.mc_train = pd.read_hdf(f'{datasetDir}/{mcName}').sample(nEvt, random_state=random_state).reset_index(drop=True) 

    def loadTestData(self, datasetDir, dataName, mcName): 
        self.data_test = pd.read_hdf(f'{datasetDir}/{dataName}')
        self.mc_test = pd.read_hdf(f'{datasetDir}/{mcName}')

    def transform(self, trans_vars, transformer, applyOn='both'): 
        if applyOn in ['train', 'both']: 
            self.data_train.loc[:, trans_vars] = transform(self.data_train.loc[:, trans_vars], transformer, trans_vars)
            self.mc_train.loc[:, trans_vars] = transform(self.mc_train.loc[:, trans_vars], transformer, trans_vars)
        elif applyOn in ['test', 'both']: 
            self.data_test.loc[:, trans_vars] = transform(self.data_test.loc[:, trans_vars], transformer, trans_vars)
            self.mc_test.loc[:, trans_vars] = transform(self.mc_test.loc[:, trans_vars], transformer, trans_vars)
        else: 
            raise NameError(f"argument 'applyOn' only accept 'train', 'test', or 'both', recieved: {applyOn}")

    def setupNN(self, num_hidden_layers, num_connected_layers, num_units, activations, dropout=None, gauss_std=None):
        self.num_hidden_layers = num_hidden_layers
        self.num_connected_layers = num_connected_layers
        self.num_units = num_units
        self.acts = activations
        self.dropout = dropout
        self.gauss_std = gauss_std

    def trainData(self, , *args, **kwargs): 
        pass

    def trainMC(self, *args, **kwargs): 
        pass

    def Correct(self, ): 
        pass

    def trainFinal(self, *args, **kwargs): 
        pass

    def drawTrainingHistories(self, ):
        pass



