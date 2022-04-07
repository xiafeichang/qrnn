import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from qrnn import trainQuantile, predict, scale
from mylib.Corrector import Corrector, applyCorrection
from mylib.transformer import fit_power_transformer, fit_quantile_transformer, transform, inverse_transform


def draw_mean_plot(x_vars, x_title, x_var_name, target, transformer_file):

    var_corr_mean = np.zeros(len(x_vars)-1)
    var_uncorr_mean = np.zeros(len(x_vars)-1)
    var_data_mean = np.zeros(len(x_vars)-1)
    x_vars_c = np.zeros(len(x_vars)-1)
    for i in range(len(x_vars)-1):
        query_str_pT = x_var_name + ' > ' + str(x_vars[i]) + ' and ' + x_var_name +' < ' + str(x_vars[i+1])

#        var_corr_mean[i] = np.mean(inverse_transform((df_mc.query(query_str_pT))['{}_corr'.format(target)], transformer_file, target)) 
#        var_uncorr_mean[i] = np.mean(inverse_transform((df_mc.query(query_str_pT))[target], transformer_file, target))
#        var_data_mean[i] = np.mean(inverse_transform((df_data.query(query_str_pT))[target], transformer_file, target))

        var_corr_mean[i] = np.mean((df_mc.query(query_str_pT))['{}_corr'.format(target)]) 
        var_uncorr_mean[i] = np.mean((df_mc.query(query_str_pT))[target])
        var_data_mean[i] = np.mean((df_data.query(query_str_pT))[target])

        x_vars_c[i] = ((x_vars[i] + x_vars[i+1])/2.)

    x_vars_c = inverse_transform(x_vars_c, transformer_file, x_var_name)
    
    fig = plt.figure(tight_layout=True)
    plt.plot(x_vars_c, var_corr_mean, label='MC corrected')
    plt.plot(x_vars_c, var_uncorr_mean, label='MC uncorrected')
    plt.plot(x_vars_c, var_data_mean, label='data')
    plt.xlabel(x_title)
    plt.ylabel('mean of {}'.format(target))
    plt.legend()
    fig.savefig('plots/check_correction/{}_{}_{}_mean.png'.format(EBEE, target, x_var_name))
    plt.close(fig)

def draw_dist_plot(qs, x_vars, x_title, x_var_name, target, transformer_file):

    nq = len(qs)
    var_corr = np.array([]).reshape(-1,nq)
    var_uncorr = np.array([]).reshape(-1,nq)
    var_data = np.array([]).reshape(-1,nq)
    x_vars_c = np.zeros(len(x_vars)-1)
    for i in range(len(x_vars)-1):
        query_str_pT = x_var_name + ' > ' + str(x_vars[i]) + ' and ' + x_var_name +' < ' + str(x_vars[i+1])

#        var_corr_mean[i] = np.mean(inverse_transform((df_mc.query(query_str_pT))['{}_corr'.format(target)], transformer_file, target)) 
#        var_uncorr_mean[i] = np.mean(inverse_transform((df_mc.query(query_str_pT))[target], transformer_file, target))
#        var_data_mean[i] = np.mean(inverse_transform((df_data.query(query_str_pT))[target], transformer_file, target))

        var_corr = np.append(var_corr, np.quantile((df_mc.query(query_str_pT))['{}_corr'.format(target)], qs).reshape(-1,nq), axis=0) 
        var_uncorr = np.append(var_uncorr, np.quantile((df_mc.query(query_str_pT))[target], qs).reshape(-1,nq), axis=0)
        var_data = np.append(var_data, np.quantile((df_data.query(query_str_pT))[target], qs).reshape(-1,nq), axis=0)

        x_vars_c[i] = ((x_vars[i] + x_vars[i+1])/2.)

    x_vars_c = inverse_transform(x_vars_c, transformer_file, x_var_name)

    var_corr = var_corr.T
    var_uncorr = var_uncorr.T
    var_data = var_data.T
    
    fig = plt.figure(tight_layout=True)
    for i in range(nq//2): 
        plt.fill_between(x_vars_c, var_corr[i], var_corr[-(i+1)], color='blue', alpha=0.1*(i+1))
        plt.fill_between(x_vars_c, var_uncorr[i], var_uncorr[-(i+1)], color='red', alpha=0.1*(i+1))
        plt.fill_between(x_vars_c, var_data[i], var_data[-(i+1)], color='grey', alpha=0.1*(i+1))

    if nq%2 != 0: 
        plt.plot(x_vars_c, var_corr[nq//2], color='blue', label='MC corrected')
        plt.plot(x_vars_c, var_uncorr[nq//2], color='red', label='MC uncorrected')
        plt.plot(x_vars_c, var_data[nq//2], color='black', label='data')
        plt.legend()

    plt.xlabel(x_title)
    plt.ylabel(target)
    fig.savefig('plots/check_correction/{}_{}_{}_dist.png'.format(EBEE, target, x_var_name))
    plt.close(fig)

variables = ['probeS4','probeR9','probeCovarianceIeIp','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
kinrho = ['probePt','probeScEta','probePhi','rho'] 

EBEE = 'EB'
nEvnt = 3000000

df_data = (pd.read_hdf('df_data_{}_test.h5'.format(EBEE))).loc[:,kinrho+variables].sample(nEvnt, random_state=100).reset_index(drop=True)
df_mc = (pd.read_hdf('df_mc_{}_test.h5'.format(EBEE))).loc[:,kinrho+variables].sample(nEvnt, random_state=100).reset_index(drop=True)

transformer_file = 'data_{}'.format(EBEE)
df_mc = transform(df_mc, transformer_file, kinrho+variables)
df_data = transform(df_data, transformer_file, kinrho+variables)

# correct
#target = variables[0]
for target in variables:
    target_raw = target[:target.find('_')] if '_' in target else target
    features = kinrho# + ['{}_corr'.format(x) for x in variables[:variables.index(target_raw)]]
    
    X = df_mc.loc[:,features]
    Y = np.array(df_mc.loc[:,target])
    
    models_mc = 'models/{}_{}_{}'.format('mc', EBEE, target)
    models_d = 'models/{}_{}_{}'.format('data', EBEE, target)
    df_mc['{}_corr'.format(target_raw)] = applyCorrection(models_mc, models_d, X, Y, diz=False)
     
    
    #pTs_raw = np.array([25., 30., 32.5, 35., 37.5, 40., 42.5, 45., 50., 60., 150.])
    pTs_raw = np.arange(25., 55., 1.5)
    pTs = transform(pTs_raw, transformer_file, 'probePt')
    etas_raw = np.arange(-1.45, 1.45, 0.15)
    etas = transform(etas_raw, transformer_file, 'probeScEta')
    #rhos_raw = np.array([0., 8., 12., 15., 18., 21., 24., 27., 30., 36., 60.])
    rhos_raw = np.arange(0., 50., 2.)
    rhos = transform(rhos_raw, transformer_file, 'rho')
    phis_raw = np.arange(-3.15, 3.15, 0.3)
    phis = transform(phis_raw, transformer_file, 'probePhi')
    
    matplotlib.use('agg')
    
    draw_mean_plot(pTs, '$p_T$', 'probePt', target, transformer_file)
    draw_mean_plot(etas, '$\eta$', 'probeScEta', target, transformer_file)
    draw_mean_plot(rhos, '$\\rho$', 'rho', target, transformer_file)
    draw_mean_plot(phis, '$\phi$', 'probePhi', target, transformer_file)
    
    qs = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
    draw_dist_plot(qs, pTs, '$p_T$', 'probePt', target, transformer_file)
    draw_dist_plot(qs, etas, '$\eta$', 'probeScEta', target, transformer_file)
    draw_dist_plot(qs, rhos, '$\\rho$', 'rho', target, transformer_file)
    draw_dist_plot(qs, phis, '$\phi$', 'probePhi', target, transformer_file)



