import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

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
gs = GridSpec(3, 1, figure=fig)
ax1 = fig.add_subplot(gs[:-1, :])
ax2 = fig.add_subplot(gs[-1, :])

histrange = (-0.0001, 0.0001)
nbin = 75
mc_uncorr_n, _, _ = ax1.hist(df_mc[target]*target_scale_par['sigma']+target_scale_par['mu'], range=histrange, bins=nbin, density=True, histtype='step', color='red', label='MC uncorrected')
mc_corr_n, _, _ = ax1.hist(df_mc['{}_corr'.format(target_raw)]*target_scale_par['sigma']+target_scale_par['mu'], range=histrange, bins=nbin, density=True, histtype='step', color='blue', label='MC corrected')

data_n, bin_edges = np.histogram(df_data[target], range=histrange, bins=nbin, density=True)
nEvnt = len(df_data)
x = np.array([])
xerr = np.array([])
for i in range(len(data_n)):
    x = np.append(x, (bin_edges[i+1]+bin_edges[i])/2.)
    xerr = np.append(xerr, (bin_edges[i+1]-bin_edges[i])/2.)
data_nerr = np.sqrt(data_n*xerr*2./nEvnt)
ax1.errorbar(x, data_n, data_nerr, xerr, fmt='.', elinewidth=1., capsize=1., color='black', label='data')

ax1.set_xlim(histrange)
ax1.set_xticks(np.arange(-0.0001,0.0001,0.00002), labels=[])
ax1.ticklabel_format(style='sci', scilimits=(-2, 3), axis='y')
ax1.legend()

ratio_uncorr = mc_uncorr_n / data_n
ratio_uncorr_err = np.sqrt((mc_uncorr_n*xerr*2./nEvnt) + (mc_uncorr_n**2/data_n)*(xerr*2./nEvnt)) / data_n
ratio_corr = mc_corr_n / data_n
ratio_corr_err = np.sqrt((mc_corr_n*xerr*2./nEvnt) + (mc_corr_n**2/data_n)*(xerr*2./nEvnt)) / data_n

ax2.plot(x, np.ones_like(x), 'k-.')
ax2.errorbar(x, ratio_uncorr, ratio_uncorr_err, xerr, fmt='.', elinewidth=1., capsize=1., color='red')
ax2.errorbar(x, ratio_corr, ratio_corr_err, xerr, fmt='.', elinewidth=1., capsize=1., color='blue')

ax2.set_xlim(histrange)
ax2.set_xticks(np.arange(-0.0001,0.0001,0.00002))
ax2.ticklabel_format(style='sci', scilimits=(-2, 3), axis='both')

plt.title(target)
fig.savefig('plots/data_mc_dist_{}'.format(target))
plt.close(fig)

