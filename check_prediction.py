import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from qrnn import trainQuantile, predict, scale
from Corrector import Corrector, applyCorrection
from transformer import fit_power_transformer, fit_quantile_transformer, transform, inverse_transform



variables = ['probeS4','probeR9','probeCovarianceIeIp','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
kinrho = ['probePt','probeScEta','probePhi','rho'] 


data_key = 'data'
EBEE = 'EB'
nEvnt = 1000000

df = (pd.read_hdf('df_{}_{}_train.h5'.format(data_key, EBEE))).loc[:,kinrho+variables].sample(nEvnt, random_state=100).reset_index(drop=True)

scale_file = 'scale_para/data_{}.h5'.format(EBEE)
scale_par = pd.read_hdf(scale_file)
transformer_file = 'data_{}'.format(EBEE)
df.loc[:,variables] = transform(df.loc[:,variables], transformer_file, variables)
df.loc[:,kinrho] = scale(df.loc[:,kinrho], scale_file=scale_file)


histranges = []

# correct
target = variables[0]
#for target, histrange in zip(variables, histranges): 
#for target in variables: 
target_raw = target[:target.find('_')] if '_' in target else target
features = kinrho# + ['{}_corr'.format(x) for x in variables[:variables.index(target_raw)]]

X = df.loc[:,features]
Y = df.loc[:,target]

print(X)
print(Y)
 
qs = np.array([0.5])
qweights = np.ones_like(qs)
model_from = 'combined_models/{}_{}_{}'.format(data_key, EBEE, target)
df['{}_pred'.format(target_raw)] = predict(X, qs, qweights, model_from)
 
target_scale_par = scale_par.loc[:,target]
matplotlib.use('agg')
fig = plt.figure(tight_layout=True)

histrange = (-4., 4.)
nbin = 75

pTs_raw = np.arange(25., 55., 1.5)
pTs = (pTs_raw - scale_par['probePt']['mu'])/scale_par['probePt']['sigma']
etas_raw = np.arange(-1.45, 1.45, 0.15)
etas = (etas_raw - scale_par['probeScEta']['mu'])/scale_par['probeScEta']['sigma']
ith = 1
df_binned = df.query('probePt > '+str(pTs[ith])+' and probePt < ' + str(pTs[ith+1]))

plt.hist(df_binned['{}_pred'.format(target_raw)], range=histrange, bins=nbin, density=True, histtype='step', label=f'pT bin {ith}')

ith = 10
df_binned = df.query('probePt > '+str(pTs[ith])+' and probePt < ' + str(pTs[ith+1]))

plt.hist(df_binned['{}_pred'.format(target_raw)], range=histrange, bins=nbin, density=True, histtype='step', label=f'pT bin {ith}')

plt.xlabel(target)
plt.legend()
fig.savefig('plots/check_pred_{}_{}_{}'.format(data_key, EBEE, target))
plt.close(fig)

