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



