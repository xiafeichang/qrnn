import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from qrnn import trainQuantile, predict, scale
from transformer import fit_power_transformer, fit_quantile_transformer, transform, inverse_transform


def draw_plot(x_vars, x_title, x_var_name, target, scale_par):

    var_pred_mean = np.zeros(len(x_vars)-1)
    var_true_mean = np.zeros(len(x_vars)-1)
    x_vars_c = np.zeros(len(x_vars)-1)
    for i in range(len(x_vars)-1):
        query_str = x_var_name + ' > ' + str(x_vars[i]) + ' and ' + x_var_name +' < ' + str(x_vars[i+1])

#        var_pred_mean[i] = np.mean(inverse_transform((df.query(query_str))[f'{target}_pred'], transformer_file, target))
#        var_true_mean[i] = np.mean(inverse_transform((df.query(query_str))[target], transformer_file, target))

        var_pred_mean[i] = np.mean((df.query(query_str))[f'{target}_pred'])
        var_true_mean[i] = np.median((df.query(query_str))[target])

        x_vars_c[i] = ((x_vars[i] + x_vars[i+1])/2.)*scale_par[x_var_name]['sigma']+scale_par[x_var_name]['mu']
    
    fig = plt.figure(tight_layout=True)
    plt.plot(x_vars_c, var_pred_mean, label='predicted')
    plt.plot(x_vars_c, var_true_mean, label='true')
    plt.xlabel(x_title)
    plt.ylabel('mean of {}'.format(target))
    plt.legend()
    fig.savefig('plots/check_{}_{}_{}_{}.png'.format(data_key, EBEE, target, x_var_name))
    plt.close(fig)

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

# correct
target = variables[0]
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



