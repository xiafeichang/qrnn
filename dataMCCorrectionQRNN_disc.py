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
from dataMCCorrectionQRNN import quantileRegressionNeuralNet
from clf_Iso import trainClfp0t, trainClf3Cat
from mylib.transformer import transform, inverse_transform
from mylib.IdMVAComputer import helpComputeIdMva
from mylib.tools import *


class quantileRegressionNeuralNetDisc(quantileRegressionNeuralNet): 

