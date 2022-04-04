import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from qrnn import trainQuantile, predict, scale
from lib.Corrector import Corrector, applyCorrection
from lib.transformer import fit_power_transformer, fit_quantile_transformer, transform, inverse_transform


def draw_plot(x_vars, x_title, x_var_name, target, scale_par):

    var_corr_mean = np.zeros(len(x_vars)-1)
    var_uncorr_mean = np.zeros(len(x_vars)-1)
    var_data_mean = np.zeros(len(x_vars)-1)
    x_vars_c = np.zeros(len(x_vars)-1)
    for i in range(len(x_vars)-1):
        query_str_pT = x_var_name + ' > ' + str(x_vars[i]) + ' and ' + x_var_name +' < ' + str(x_vars[i+1])

        var_corr_mean[i] = np.mean(inverse_transform((df_mc.query(query_str_pT))['{}_corr'.format(target)], transformer_file, target)) 
        var_uncorr_mean[i] = np.mean(inverse_transform((df_mc.query(query_str_pT))[target], transformer_file, target))
        var_data_mean[i] = np.mean(inverse_transform((df_data.query(query_str_pT))[target], transformer_file, target))

#        var_corr_mean[i] = np.mean((df_mc.query(query_str_pT))['{}_corr'.format(target)]) 
#        var_uncorr_mean[i] = np.mean((df_mc.query(query_str_pT))[target])
#        var_data_mean[i] = np.mean((df_data.query(query_str_pT))[target])

        x_vars_c[i] = ((x_vars[i] + x_vars[i+1])/2.)*scale_par[x_var_name]['sigma']+scale_par[x_var_name]['mu']
    
    fig = plt.figure(tight_layout=True)
    plt.plot(x_vars_c, var_corr_mean, label='MC corrected')
    plt.plot(x_vars_c, var_uncorr_mean, label='MC uncorrected')
    plt.plot(x_vars_c, var_data_mean, label='data')
    plt.xlabel(x_title)
    plt.ylabel('mean of {}'.format(target))
    plt.legend()
    fig.savefig('plots/check_result_{}_{}_{}.png'.format(EBEE, target, x_var_name))
    plt.close(fig)

variables = ['probeS4','probeR9','probeCovarianceIeIp','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
kinrho = ['probePt','probeScEta','probePhi','rho'] 

EBEE = 'EB'
nEvnt = 1000000

df_data = (pd.read_hdf('df_data_{}_test.h5'.format(EBEE))).loc[:,kinrho+variables].sample(nEvnt, random_state=100).reset_index(drop=True)
df_mc = (pd.read_hdf('df_mc_{}_test.h5'.format(EBEE))).loc[:,kinrho+variables].sample(nEvnt, random_state=100).reset_index(drop=True)

scale_file = 'scale_para/data_{}.h5'.format(EBEE)
scale_par = pd.read_hdf(scale_file)
transformer_file = 'data_{}'.format(EBEE)
df_mc.loc[:,variables] = transform(df_mc.loc[:,variables], transformer_file, variables)
df_mc.loc[:,kinrho] = scale(df_mc.loc[:,kinrho], scale_file=scale_file)

df_data.loc[:,variables] = transform(df_data.loc[:,variables], transformer_file, variables)
df_data.loc[:,kinrho] = scale(df_data.loc[:,kinrho], scale_file=scale_file)

# correct
target = variables[0]
#for target in variables:
target_raw = target[:target.find('_')] if '_' in target else target
features = kinrho# + ['{}_corr'.format(x) for x in variables[:variables.index(target_raw)]]

X = df_mc.loc[:,features]
Y = np.array(df_mc.loc[:,target])

print(X)
print(Y)
 
models_mc = 'combined_models/{}_{}_{}'.format('mc', EBEE, target)
models_d = 'combined_models/{}_{}_{}'.format('data', EBEE, target)
df_mc['{}_corr'.format(target_raw)] = applyCorrection(models_mc, models_d, X, Y, diz=False)
 

#pTs_raw = np.array([25., 30., 32.5, 35., 37.5, 40., 42.5, 45., 50., 60., 150.])
pTs_raw = np.arange(25., 55., 1.5)
pTs = (pTs_raw - scale_par['probePt']['mu'])/scale_par['probePt']['sigma']
etas_raw = np.arange(-1.45, 1.45, 0.15)
etas = (etas_raw - scale_par['probeScEta']['mu'])/scale_par['probeScEta']['sigma']
#rhos_raw = np.array([0., 8., 12., 15., 18., 21., 24., 27., 30., 36., 60.])
rhos_raw = np.arange(0., 50., 2.)
rhos = (rhos_raw - scale_par['rho']['mu'])/scale_par['rho']['sigma']
phis_raw = np.arange(-3.15, 3.15, 0.3)
phis = (phis_raw - scale_par['probePhi']['mu'])/scale_par['probePhi']['sigma']

matplotlib.use('agg')

draw_plot(pTs, '$p_T$', 'probePt', target, scale_par)
draw_plot(etas, '$\eta$', 'probeScEta', target, scale_par)
draw_plot(rhos, '$\\rho$', 'rho', target, scale_par)
draw_plot(phis, '$\phi$', 'probePhi', target, scale_par)



