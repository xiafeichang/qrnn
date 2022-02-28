import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from qrnn import trainQuantile, predict, scale
from Corrector import Corrector, applyCorrection



variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
kinrho = ['probePt','probeScEta','probePhi','rho'] 

df_data = (pd.read_hdf('df_data_EB_test.h5')).loc[:,kinrho+variables]#.sample(2000, random_state=100).reset_index(drop=True)
df_mc_raw = (pd.read_hdf('df_mc_EB_test.h5')).loc[:,kinrho+variables]#.sample(2000, random_state=100).reset_index(drop=True)

scale_file = 'scale_para/data.h5'
scale_par = pd.read_hdf(scale_file)
df_mc = scale(df_mc_raw, scale_file=scale_file) 

# correct
target = variables[0]
target_raw = target[:target.find('_')] if '_' in target else target
features = kinrho + ['{}_corr'.format(x) for x in variables[:variables.index(target_raw)]]

X = df_mc.loc[:,features]
Y = df_mc.loc[:,target]
 
models_mc = 'combined_models/{}_{}'.format('mc', target)
models_d = 'combined_models/{}_{}'.format('data', target)
df_mc['{}_corr'.format(target_raw)] = applyCorrection(models_mc, models_d, X, Y, diz=False)
 

target_scale_par = scale_par.loc[:,target]
matplotlib.use('agg')
fig = plt.figure(tight_layout=True)
plt.hist(df_mc[target]*target_scale_par['sigma']+target_scale_par['mu'], range=(-0.0001, 0.0001), bins=100, density=True, histtype='step', label='MC uncorrected')
plt.hist(df_mc['{}_corr'.format(target_raw)]*target_scale_par['sigma']+target_scale_par['mu'], range=(-0.0001, 0.0001), bins=100, density=True, histtype='step', label='MC corrected')
plt.hist(df_data[target], range=(-0.0001, 0.0001), bins=100, density=True, histtype='step', label='data')
plt.legend()
fig.savefig('plots/data_mc_dist_{}'.format(target))
plt.close(fig)

