import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
#matplotlib.use('pdf')
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import xgboost as xgb
from sklearn.utils import shuffle
from time import time
import pickle
import gzip
from joblib import delayed, Parallel, parallel_backend, register_parallel_backend

from qrnn import trainQuantile, predict, scale
from mylib.Corrector import Corrector, applyCorrection
from mylib.Shifter import Shifter, applyShift
from mylib.Shifter2D import Shifter2D, apply2DShift
from mylib.transformer import transform, inverse_transform
from mylib.IdMVAComputer import helpComputeIdMva
from mylib.tools import *


def clf_reweight(df_mc, df_data, clf_name, n_jobs=1, cut=None):

    features = ['probePt','probeScEta','probePhi','rho']
    try:
        clf = pickle.load(open(f"{clf_name}.pkl", "rb"))
        print("Loaded classifier from file {}.pkl".format(clf_name))
    except FileNotFoundError:
        print('Training classifier')
        clf = xgb.XGBClassifier(learning_rate=0.05,n_estimators=500,max_depth=10,gamma=0,n_jobs=n_jobs)
        if cut is not None:
            X_data = df_data.query(cut, engine='python').sample(min(min(df_mc.query(cut, engine='python').index.size,df_data.query(cut, engine='python').index.size), 1000000)).loc[:,features].values
            X_mc = df_mc.query(cut, engine='python').sample(min(min(df_mc.query(cut, engine='python').index.size,df_data.query(cut, engine='python').index.size), 1000000)).loc[:,features].values
        else:
            X_data = df_data.sample(min(min(df_mc.index.size,df_data.index.size), 1000000)).loc[:,features].values
            X_mc = df_mc.sample(min(min(df_mc.index.size,df_data.index.size), 1000000)).loc[:,features].values
        X = np.vstack([X_data,X_mc])
        y = np.vstack([np.ones((X_data.shape[0],1)),np.zeros((X_mc.shape[0],1))])
        X, y = shuffle(X,y)

        start = time()
        clf.fit(X,y)
        print("Classifier trained in {:.2f} seconds".format(time() - start))
        with open(f"{clf_name}.pkl", "wb") as f:
            pickle.dump(clf, f)
    eps = 1.e-3
    return np.apply_along_axis(lambda x: x[1]/(x[0]+eps), 1, clf.predict_proba(df_mc.loc[:,features].values))

def draw_mean_plot(EBEE, df_data, df_mc, x_vars, x_title, x_var_name, target, plotsdir):

    var_corr_mean = np.zeros(len(x_vars)-1)
    var_uncorr_mean = np.zeros(len(x_vars)-1)
    var_data_mean = np.zeros(len(x_vars)-1)
    x_vars_c = np.zeros(len(x_vars)-1)
    for i in range(len(x_vars)-1):
        query_str = x_var_name + ' > ' + str(x_vars[i]) + ' and ' + x_var_name +' < ' + str(x_vars[i+1])

        var_corr_mean[i] = np.mean((df_mc.query(query_str))['{}_corr'.format(target)]) 
        var_uncorr_mean[i] = np.mean((df_mc.query(query_str))[target])
        var_data_mean[i] = np.mean((df_data.query(query_str))[target])

        x_vars_c[i] = ((x_vars[i] + x_vars[i+1])/2.)

    if EBEE == 'EE' and x_var_name == 'probeScEta': 
        idx_del = np.asarray(np.abs(x_vars_c)<1.5).nonzero()
        var_corr_mean = np.delete(var_corr_mean, idx_del, axis=0)
        var_uncorr_mean = np.delete(var_uncorr_mean, idx_del, axis=0)
        var_data_mean = np.delete(var_data_mean, idx_del, axis=0)

        x_vars_c = np.delete(x_vars_c, idx_del, axis=0)

    fig = plt.figure(tight_layout=True)
    plt.plot(x_vars_c, var_corr_mean,   color='green', label='MC corrected')
    plt.plot(x_vars_c, var_uncorr_mean, color='red',   label='MC uncorrected')
    plt.plot(x_vars_c, var_data_mean,   color='blue',  label='data')

    plt.title(r'$\bf{CMS}$ $\it{Work\ in\ Progress}$', loc='left')
    plt.title(f'UL 2018 {EBEE}', loc='right')
    plt.xlabel(x_title)
    plt.ylabel('mean of {}'.format(target))
    plt.legend()
    fig.savefig('{}/{}_{}_{}_mean.png'.format(plotsdir, EBEE, target, x_var_name))
#    fig.savefig('{}/{}_{}_{}_mean.pdf'.format(plotsdir, EBEE, target, x_var_name))
    plt.close(fig)

def draw_dist_plot(EBEE, df_data, df_mc, qs, x_vars, x_title, x_var_name, target, plotsdir):

    nq = len(qs)
    var_corr = np.array([]).reshape(-1,nq)
    var_uncorr = np.array([]).reshape(-1,nq)
    var_data = np.array([]).reshape(-1,nq)
    x_vars_c = np.zeros(len(x_vars)-1)
    for i in range(len(x_vars)-1):
        query_str = x_var_name + ' > ' + str(x_vars[i]) + ' and ' + x_var_name +' < ' + str(x_vars[i+1])

        var_corr = np.append(var_corr, np.quantile((df_mc.query(query_str))['{}_corr'.format(target)], qs).reshape(-1,nq), axis=0) 
        var_uncorr = np.append(var_uncorr, np.quantile((df_mc.query(query_str))[target], qs).reshape(-1,nq), axis=0)
        var_data = np.append(var_data, np.quantile((df_data.query(query_str))[target], qs).reshape(-1,nq), axis=0)

        x_vars_c[i] = ((x_vars[i] + x_vars[i+1])/2.)

    if EBEE == 'EE' and x_var_name == 'probeScEta': 
        idx_del = np.asarray(np.abs(x_vars_c)<1.5).nonzero()
        var_corr = np.delete(var_corr, idx_del, axis=0)
        var_uncorr = np.delete(var_uncorr, idx_del, axis=0)
        var_data = np.delete(var_data, idx_del, axis=0)

        x_vars_c = np.delete(x_vars_c, idx_del, axis=0)

    var_corr = var_corr.T
    var_uncorr = var_uncorr.T
    var_data = var_data.T
    
    fig = plt.figure(tight_layout=True)

    if nq > 1: 
        plt.plot(x_vars_c, var_corr[0],   linestyle='--', color='green')
        plt.plot(x_vars_c, var_uncorr[0], linestyle='--', color='red')
        plt.plot(x_vars_c, var_data[0],   linestyle='--', color='blue')

        plt.plot(x_vars_c, var_corr[-1],   linestyle='--', color='green')
        plt.plot(x_vars_c, var_uncorr[-1], linestyle='--', color='red')
        plt.plot(x_vars_c, var_data[-1],   linestyle='--', color='blue')

        for i in range(1,nq//2): 
            plt.fill_between(x_vars_c, var_corr[i],   var_corr[-(i+1)],   color='green', alpha=0.1*(i+1))
            plt.fill_between(x_vars_c, var_uncorr[i], var_uncorr[-(i+1)], color='red',   alpha=0.1*(i+1))
            plt.fill_between(x_vars_c, var_data[i],   var_data[-(i+1)],   color='blue',  alpha=0.1*(i+1))

    if nq%2 != 0: 
        plt.plot(x_vars_c, var_corr[nq//2],   color='green', label='MC corrected')
        plt.plot(x_vars_c, var_uncorr[nq//2], color='red',   label='MC uncorrected')
        plt.plot(x_vars_c, var_data[nq//2],   color='blue',  label='data')
        plt.legend()

    plt.title(r'$\bf{CMS}$ $\it{Work\ in\ Progress}$', loc='left')
    plt.title(f'UL 2018 {EBEE}', loc='right')
    plt.xlabel(x_title)
    plt.ylabel(target)
    fig.savefig('{}/{}_{}_{}_dist.png'.format(plotsdir, EBEE, target, x_var_name))
#    fig.savefig('{}/{}_{}_{}_dist.pdf'.format(plotsdir, EBEE, target, x_var_name))
    plt.close(fig)

def draw_hist(df_data, df_mc, nEvt, target, fig_name, bins=None, histrange=None, density=False, mc_weights=None, logplot=False):

    if 'EB' in fig_name: 
        EBEE = 'EB'
    elif 'EE' in fig_name: 
        EBEE = 'EE'
    else: 
        EBEE = 'all'
    
    mc_weights = mc_weights * nEvt / np.sum(mc_weights)
#    mc_weights = None
    
    fig = plt.figure(tight_layout=True)
    gs = GridSpec(3, 1, figure=fig)
    ax1 = fig.add_subplot(gs[:-1, :])
    ax2 = fig.add_subplot(gs[-1, :])
    
    if 'Iso' in target: 
        mc_shift_n, _, _ = ax1.hist(df_mc['{}_shift'.format(target)], range=histrange, bins=bins, density=density, weights=mc_weights, histtype='step', color='green', label='MC shifted')

    mc_uncorr_n, _, _ = ax1.hist(df_mc[target], range=histrange, bins=bins, density=density, weights=mc_weights, histtype='step', color='red', label='MC uncorrected')
    mc_corr_n, _, _ = ax1.hist(df_mc['{}_corr'.format(target)], range=histrange, bins=bins, density=density, weights=mc_weights, histtype='step', color='blue', label='MC corrected')
    data_n, bin_edges = np.histogram(df_data[target], range=histrange, bins=bins, density=density)
    
    x = np.array([])
    xerr = np.array([])
    for i in range(len(data_n)):
        x = np.append(x, (bin_edges[i+1]+bin_edges[i])/2.)
        xerr = np.append(xerr, (bin_edges[i+1]-bin_edges[i])/2.)
    data_nerr = np.sqrt(data_n)
#    data_nerr = np.sqrt(data_n*xerr*2./nEvt)
    ax1.errorbar(x, data_n, data_nerr, xerr, fmt='.', elinewidth=1., capsize=1., color='black', label='data')
    
    xticks = np.linspace(histrange[0],histrange[1],10)
    ax1.set_xlim(histrange)
    ax1.set_xticks(xticks, labels=[])
#    ax1.ticklabel_format(style='sci', scilimits=(-2, 3), axis='y')
    ax1.legend()
    ax1.set_title(r'$\bf{CMS}$ $\it{Work\ in\ Progress}$', loc='left')
    ax1.set_title(f'UL 2018 {EBEE}', loc='right') # fontsize=16, fontname="Times New Roman"
    ax1.set_ylabel(f'Events / {(histrange[1]-histrange[0])/bins}')
    if logplot: 
        ax1.set_yscale('log')
    
    ratio_uncorr = data_n / mc_uncorr_n
    ratio_uncorr_err = np.sqrt(data_n + (data_n**2/mc_uncorr_n)) / mc_uncorr_n
#    ratio_uncorr_err = np.sqrt((data_n*xerr*2./nEvt) + (data_n**2/mc_uncorr_n)*(xerr*2./nEvt)) / mc_uncorr_n
    ratio_corr = data_n / mc_corr_n
    ratio_corr_err = np.sqrt(data_n + (data_n**2/mc_corr_n)) / mc_corr_n
#    ratio_corr_err = np.sqrt((data_n*xerr*2./nEvt) + (data_n**2/mc_corr_n)*(xerr*2./nEvt)) / mc_corr_n

#    ratio_uncorr = (mc_uncorr_n - data_n) / data_n
#    ratio_uncorr_err = np.sqrt(((mc_uncorr_n+data_n)*xerr*2./nEvt) + ((mc_uncorr_n-data_n)**2/data_n)*(xerr*2./nEvt)) / data_n
#    ratio_corr = (mc_corr_n - data_n) / data_n
#    ratio_corr_err = np.sqrt(((mc_corr_n+data_n)*xerr*2./nEvt) + ((mc_corr_n-data_n)**2/data_n)*(xerr*2./nEvt)) / data_n
    
    ax2.plot(x, np.ones_like(x), 'k-.')
#    ax2.plot(x, np.zeros_like(x), 'k-.')
    ax2.errorbar(x, ratio_uncorr, ratio_uncorr_err, xerr, fmt='.', elinewidth=1., capsize=1., color='red')

#    if 'Iso' in target: 
#        ratio_shift = data_n / mc_shift_n
#        ratio_shift_err = np.sqrt(data_n + (data_n**2/mc_shift_n)) / mc_shift_n
##        ratio_shift_err = np.sqrt((data_n*xerr*2./nEvt) + (data_n**2/mc_shift_n)*(xerr*2./nEvt)) / mc_shift_n
#        ax2.errorbar(x, ratio_shift, ratio_shift_err, xerr, fmt='.', elinewidth=1., capsize=1., color='green')

    ax2.errorbar(x, ratio_corr, ratio_corr_err, xerr, fmt='.', elinewidth=1., capsize=1., color='blue')
    
    ax2.grid(True)
    ax2.set_xlim(histrange)
    ax2.set_ylim(0.8, 1.2)
#    ax2.set_ylim(-0.2, 0.2)
    ax2.set_xticks(xticks)
    ax2.ticklabel_format(style='sci', scilimits=(-2, 3), axis='both')
    ax2.set_ylabel('Data / MC')
    ax2.set_xlabel(target)
    
    fig.savefig(f'{fig_name}.png')
#    fig.savefig(f'{fig_name}.pdf')
    plt.close(fig)



def main(options):
    variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    isoVarsPh = ['probePhoIso']
    isoVarsCh = ['probeChIso03','probeChIso03worst']
    preshower = ['probeesEnergyOverSCRawEnergy']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    
    EBEE = options.EBEE
#    nEvt = 3500000
    nEvt = options.nEvt

#    df_data = (pd.read_hdf('dataframes/df_data_{}_Iso_test.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dataframes/df_mc_{}_Iso_test.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)

    df_data = (pd.read_hdf('dataframes/df_data_{}_test.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dataframes/df_mc_{}_test.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
     
#    modeldir = 'chained_models'
    plotsdir = f'plots/check_correction_final/{EBEE}'

    
    
#
#    # shift isolation varaibles
#    print(f'shifting mc with classifiers and tail regressor for {isoVarsPh}')
#    with parallel_backend('loky'): 
#        Y_shifted = parallelize(applyShift,
#            df_mc.loc[:,kinrho].values, df_mc.loc[:,isoVarsPh[0]].values,
#            load_clf('{}/mc_{}_clf_{}.pkl'.format(modeldir, EBEE, isoVarsPh[0])), 
#            load_clf('{}/data_{}_clf_{}.pkl'.format(modeldir, EBEE, isoVarsPh[0])), 
#            '{}/mc_{}_{}'.format(modeldir, EBEE, isoVarsPh[0]),
#            final_reg = False,
#            ) 
#    df_mc['{}_shift'.format(isoVarsPh[0])] = Y_shifted 
#
#    print(f'shifting mc with classifiers and tail regressor for {isoVarsCh}')
#    with parallel_backend('loky'): 
#        Y_shifted2D = parallelize(apply2DShift,
#            df_mc.loc[:,kinrho].values, df_mc.loc[:,isoVarsCh].values,
#            load_clf('{}/mc_{}_clf_{}_{}.pkl'.format(modeldir, EBEE, isoVarsCh[0], isoVarsCh[1])), 
#            load_clf('{}/data_{}_clf_{}_{}.pkl'.format(modeldir, EBEE, isoVarsCh[0], isoVarsCh[1])), 
#            '{}/mc_{}_tReg_{}'.format(modeldir, EBEE, isoVarsCh[0]), 
#            '{}/mc_{}_tReg_{}'.format(modeldir, EBEE, isoVarsCh[1]), 
##            n_jobs = 3,
#            final_reg = False,
#            ) 
#    df_mc['{}_shift'.format(isoVarsCh[0])] = Y_shifted2D[:,0]
#    df_mc['{}_shift'.format(isoVarsCh[1])] = Y_shifted2D[:,1]
#
#    # transform features and target to apply qrnn
#    transformer_file = 'data_{}'.format(EBEE)
#    df_mc.loc[:,kinrho+variables] = transform(df_mc.loc[:,kinrho+variables], transformer_file, kinrho+variables)
#    #df_data.loc[:,kinrho+variables] = transform(df_data.loc[:,kinrho+variables], transformer_file, kinrho+variables)
#    print(df_mc)
#    
#
#    # correct
#    corr_start = time()
#    if EBEE == 'EB': 
#        vars_qrnn = variables.copy() 
#    else: 
#        vars_qrnn = variables+preshower
#    for target in vars_qrnn: 
#        features = kinrho + ['{}_corr'.format(x) for x in variables[:variables.index(target)]]
#        
#        X = df_mc.loc[:,features]
#        Y = df_mc.loc[:,target]
#        
#        models_mc = '{}/{}_{}_{}'.format(modeldir, 'mc', EBEE, target)
#        models_d = '{}/{}_{}_{}'.format(modeldir, 'data', EBEE, target)
#        print('Correct {} with features {}'.format(target, features))
#        df_mc['{}_corr'.format(target)] = parallelize(applyCorrection, X, Y, models_mc, models_d, diz=False)
#    
#    
#
#    print('Correct {} with features {}'.format(isoVarsPh[0], kinrho))
#    df_mc['{}_corr'.format(isoVarsPh[0])] = parallelize(applyCorrection,
#        df_mc.loc[:,kinrho], df_mc.loc[:,'{}_shift'.format(isoVarsPh[0])], 
#        '{}/{}_{}_{}'.format(modeldir, 'mc', EBEE, isoVarsPh[0]), 
#        '{}/{}_{}_{}'.format(modeldir, 'data', EBEE, isoVarsPh[0]), 
#        diz=True, 
#        )
#
#
#    print(f'correcting mc with models for {isoVarsCh}')
#    for target in isoVarsCh: 
#        features = kinrho + ['{}_corr'.format(x) for x in isoVarsCh[:isoVarsCh.index(target)]]
#        
#        X = df_mc.loc[:,features]
#        Y = df_mc.loc[:,'{}_shift'.format(target)]
#        
#        models_mc = '{}/{}_{}_{}'.format(modeldir, 'mc', EBEE, target)
#        models_d = '{}/{}_{}_{}'.format(modeldir, 'data', EBEE, target)
#        print('Correct {} with features {}'.format(target, features))
#        df_mc['{}_corr'.format(target)] = parallelize(applyCorrection, X, Y, models_mc, models_d, diz=False)
# 
#
#    vars_corr = ['{}_corr'.format(target) for target in variables] 
##    df_mc.loc[:,variables+vars_corr] = inverse_transform(df_mc.loc[:,variables+vars_corr], transformer_file, variables+vars_corr)
#    df_mc.loc[:,kinrho] = inverse_transform(df_mc.loc[:,kinrho], transformer_file, kinrho)
#
#
#    if EBEE != 'EB': 
#        vars_corr = vars_corr + ['{}_corr'.format(var) for var in preshower]
#    isoVars = isoVarsPh+isoVarsCh
#    isoVars_shift = ['{}_shift'.format(var) for var in isoVars]
#    isoVars_corr = ['{}_corr'.format(var) for var in isoVars]
##    print(df_mc.loc[:,kinrho+vars_qrnn+vars_corr+isoVars+isoVars_shift+isoVars_corr])
#    print('time spent in correction: {} s'.format(time() - corr_start))
#
#    id_start = time()
#    weightsEB = 'phoIDmva_weight/HggPhoId_94X_barrel_BDT_v2.weights.xml'
#    weightsEE = 'phoIDmva_weight/HggPhoId_94X_endcap_BDT_v2.weights.xml'
#    
#    phoIDname = 'probePhoIdMVA'
#    print('Compute photon ID MVA for data')
#    df_data[phoIDname] = helpComputeIdMva(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars, df_data, 'data', False) # variables+isoVars
#    print('Compute photon ID MVA for uncorrected mc')
#    df_mc[phoIDname] = helpComputeIdMva(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars, df_mc, 'data', False) # +isoVars 
#    print('Compute photon ID MVA for corrected mc')
#    df_mc['{}_corr'.format(phoIDname)] = helpComputeIdMva(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars, df_mc, 'qr', False) # +isoVars 
#    print('time spent in computing photon ID MVA: {} s'.format(time() - id_start))
#
#    df_mc.to_hdf('test/dfs_corr/df_mc_{}_Iso_test_corr.h5'.format(EBEE),'df',mode='w',format='t')
#    print(df_mc.keys())
        

    # draw plots
    df_mc = pd.read_hdf('dfs_corr/df_mc_{}_test_corr_final.h5'.format(EBEE))
    print('Computing weights')
    df_mc['weight_clf'] = clf_reweight(df_mc, df_data, f'transformer/4d_reweighting_{EBEE}', n_jobs=10)

    if EBEE == 'EB':
        histranges = {'probeS4':(0., 1.), 
                      'probeR9':(0., 1.2), 
                      'probeCovarianceIeIp':(-2.e-4, 2.e-4), 
                      'probePhiWidth':(0., 0.2), 
                      'probeSigmaIeIe':(0., 2.e-2), 
                      'probeEtaWidth':(0., 0.05), 
                      'probePhoIso':(0., 10.), 
                      'probeChIso03':(0., 10.), 
                      'probeChIso03worst':(0., 10.)}
    else: 
        histranges = {'probeS4':(0., 1.), 
                      'probeR9':(0., 1.2), 
                      'probeCovarianceIeIp':(-1.5e-3, 1.5e-3), 
                      'probePhiWidth':(0., 0.2), 
                      'probeSigmaIeIe':(0.01, 0.05), 
                      'probeEtaWidth':(0., 0.05), 
                      'probePhoIso':(0., 10.), 
                      'probeChIso03':(0., 10.), 
                      'probeChIso03worst':(0., 10.)}

    logplots = {'probeS4'            :False, 
                'probeR9'            :False, 
                'probeCovarianceIeIp':False, 
                'probePhiWidth'      :False, 
                'probeSigmaIeIe'     :False, 
                'probeEtaWidth'      :False, 
                'probePhoIso'        :False, 
                'probeChIso03'       :True, 
                'probeChIso03worst'  :False}

    #histrange = (-4., 4.)
    bins = 100
    
    #pTs = np.array([25., 30., 32.5, 35., 37.5, 40., 42.5, 45., 50., 60., 150.])
    pTs = np.arange(25., 55., 1.5)
    if EBEE == 'EB':
        etas = np.arange(-1.45, 1.45, 0.15)
    else: 
        etas = np.append(np.arange(-2.5, -1.556, 0.2), np.arange(1.556, 2.5, 0.2))
    #rhos = np.array([0., 8., 12., 15., 18., 21., 24., 27., 30., 36., 60.])
    rhos = np.arange(0., 50., 2.)
    phis = np.arange(-3.15, 3.15, 0.3)

    xs      = [pTs,       etas,         rhos,      phis      ]
    xtitles = ['$p_T$',   '$\eta$',     '$\\rho$', '$\phi$'  ]
    xnames  = ['probePt', 'probeScEta', 'rho',     'probePhi']
    qs = np.array([0.025, 0.16, 0.5, 0.84, 0.975])

    for target in variables: #isoVars+variables+preshower 
        fig_name = '{}/data_mc_dist_{}_{}'.format(plotsdir, EBEE, target)
    
        if target in preshower: 
            query_preshower = 'probeScEta<-1.653 or probeScEta>1.653'
            draw_hist(df_data.query(query_preshower), df_mc.query(query_preshower), nEvt, target, fig_name, bins, histranges[target], mc_weights=df_mc['weight_clf'], logplot=logplots[target])
             
            for x, xtitle, xname in zip(xs, xtitles, xnames): 
                draw_mean_plot(EBEE, df_data.query(query_preshower), df_mc.query(query_preshower), x, xtitle, xname, target, plotsdir)
                draw_dist_plot(EBEE, df_data.query(query_preshower), df_mc.query(query_preshower), qs, x, xtitle, xname, target, plotsdir)
#        elif target in isoVars: 
#            draw_hist(df_data, df_mc, nEvt, target, fig_name, bins, histranges[target], mc_weights=df_mc['weight_clf'], logplot=logplots[target])
#             
#            query_iso = f'{target}!=0'
#            for x, xtitle, xname in zip(xs, xtitles, xnames): 
#                draw_mean_plot(EBEE, df_data.query(query_iso), df_mc.query(query_iso) x, xtitle, xname, target, plotsdir)
#                draw_dist_plot(EBEE, df_data.query(query_iso), df_mc.query(query_iso) qs, x, xtitle, xname, target, plotsdir)
        else: 
            draw_hist(df_data, df_mc, nEvt, target, fig_name, bins, histranges[target], mc_weights=df_mc['weight_clf'], logplot=logplots[target])
             
            for x, xtitle, xname in zip(xs, xtitles, xnames): 
                draw_mean_plot(EBEE, df_data, df_mc, x, xtitle, xname, target, plotsdir)
                draw_dist_plot(EBEE, df_data, df_mc, qs, x, xtitle, xname, target, plotsdir)

  
#    draw_hist(
#        df_data, df_mc, 
#        nEvt,
#        phoIDname, 
#        '{}/data_mc_dist_{}_{}'.format(plotsdir, EBEE, phoIDname),
#        bins = bins, 
#        histrange = (0., 1.),
#        mc_weights = df_mc['weight_clf'], 
#        )
#
#    for x, xtitle, xname in zip(xs, xtitles, xnames): 
#        draw_mean_plot(EBEE, df_data, df_mc, x, xtitle, xname, phoIDname, plotsdir)
#        draw_dist_plot(EBEE, df_data, df_mc, qs, x, xtitle, xname, phoIDname, plotsdir)




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-n','--nEvt', action='store', type=int, required=True)
    options = parser.parse_args()
    main(options)
