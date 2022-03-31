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
gs = GridSpec(3, 1, figure=fig)
ax1 = fig.add_subplot(gs[:-1, :])
ax2 = fig.add_subplot(gs[-1, :])

histrange = (-4., 4.)
nbin = 75

#mc_uncorr_n, _, _ = ax1.hist(inverse_transform(df_mc[target], transformer_file, target), range=histrange, bins=nbin, density=True, histtype='step', color='red', label='MC uncorrected')
#pred_n, _, _ = ax1.hist(inverse_transform(df_mc['{}_corr'.format(target_raw)], transformer_file, target), range=histrange, bins=nbin, density=True, histtype='step', color='blue', label='MC corrected')
#data_n, bin_edges = np.histogram(inverse_transform(df[target], transformer_file, target), range=histrange, bins=nbin, density=True)

pred_n, _, _ = ax1.hist(df['{}_pred'.format(target_raw)], range=histrange, bins=nbin, density=True, histtype='step', color='blue', label='predicted')
data_n, bin_edges = np.histogram(df[target], range=histrange, bins=nbin, density=True)

x = np.array([])
xerr = np.array([])
for i in range(len(data_n)):
    x = np.append(x, (bin_edges[i+1]+bin_edges[i])/2.)
    xerr = np.append(xerr, (bin_edges[i+1]-bin_edges[i])/2.)
data_nerr = np.sqrt(data_n*xerr*2./nEvnt)
ax1.errorbar(x, data_n, data_nerr, xerr, fmt='.', elinewidth=1., capsize=1., color='black', label='true')

xticks = np.linspace(histrange[0],histrange[1],10)
ax1.set_xlim(histrange)
ax1.set_xticks(xticks, labels=[])
ax1.ticklabel_format(style='sci', scilimits=(-2, 3), axis='y')
ax1.legend()

ratio = pred_n / data_n
ratio_err = np.sqrt((pred_n*xerr*2./nEvnt) + (pred_n**2/data_n)*(xerr*2./nEvnt)) / data_n

ax2.plot(x, np.ones_like(x), 'k-.')
ax2.errorbar(x, ratio, ratio_err, xerr, fmt='.', elinewidth=1., capsize=1., color='blue')

ax2.set_xlim(histrange)
ax2.set_ylim(0., 3.)
ax2.set_xticks(xticks)
ax2.ticklabel_format(style='sci', scilimits=(-2, 3), axis='both')

plt.title(target)
fig.savefig('plots/check_pred_{}_{}_{}'.format(data_key, EBEE, target))
plt.close(fig)

