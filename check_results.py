import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('cairo')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'Helvetica'
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from joblib import delayed, Parallel

import xgboost as xgb
from sklearn.utils import shuffle
from time import time
import pickle
import gzip

from mylib.IdMVAComputer import helpComputeIdMva
from mylib.tools import *

varNameMap = {'probeS4':r'$S_4$', 
              'probeR9':r'$R_9$', 
              'probeCovarianceIeIp':r'$Cov_{i\eta i\phi}$', 
              'probePhiWidth':r'$\phi$-width', 
              'probeSigmaIeIe':r'$\sigma_{i\eta i\eta}$', 
              'probeEtaWidth':r'$\eta$-width', 
              'probePhoIso':r'$Iso_{\gamma}$', 
              'probeChIso03':r'$Iso_{Ch}$', 
              'probeChIso03worst':r'$Iso_{ChWorst}$', 
              'probeesEnergyOverSCRawEnergy':r'$E_{es}/E_{SC}$',
              'probePhoIdMVA':'photon ID MVA score'} 


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

def draw_mean_plot(EBEE, df_data, df_mc, x_vars, x_title, x_var_name, target, plotsdir, final=False):

    var_corr_final_mean = np.zeros(len(x_vars)-1)
    var_corr_mean = np.zeros(len(x_vars)-1)
    var_uncorr_mean = np.zeros(len(x_vars)-1)
    var_data_mean = np.zeros(len(x_vars)-1)
    x_vars_c = np.zeros(len(x_vars)-1)
    for i in range(len(x_vars)-1):
        query_str = x_var_name + ' > ' + str(x_vars[i]) + ' and ' + x_var_name +' < ' + str(x_vars[i+1])
        df_mc_queried = df_mc.query(query_str)

        if final:
            var_corr_final_mean[i] = np.average(df_mc_queried['{}_corr_final'.format(target)], weights=df_mc_queried['weight_clf']) 
        var_corr_mean[i] = np.average(df_mc_queried['{}_corr'.format(target)], weights=df_mc_queried['weight_clf']) 
        var_uncorr_mean[i] = np.average(df_mc_queried[target], weights=df_mc_queried['weight_clf'])
        var_data_mean[i] = np.average((df_data.query(query_str))[target])

        x_vars_c[i] = ((x_vars[i] + x_vars[i+1])/2.)

    if EBEE == 'EE' and x_var_name == 'probeScEta': 
        idx_del = np.asarray(np.abs(x_vars_c)<1.5).nonzero()
        var_corr_mean = np.delete(var_corr_mean, idx_del, axis=0)
        var_uncorr_mean = np.delete(var_uncorr_mean, idx_del, axis=0)
        var_data_mean = np.delete(var_data_mean, idx_del, axis=0)
        if final:
            var_corr_final_mean = np.delete(var_corr_final_mean, idx_del, axis=0)

        x_vars_c = np.delete(x_vars_c, idx_del, axis=0)

    fig = plt.figure(tight_layout=True)
    plt.plot(x_vars_c, var_corr_mean,   color='green', label='MC corrected')
    plt.plot(x_vars_c, var_uncorr_mean, color='red',   label='MC uncorrected')
    plt.plot(x_vars_c, var_data_mean,   color='blue',  label='data')
    if final:
        plt.plot(x_vars_c, var_corr_final_mean, color='cyan', label='MC corrected final')

#    plt.title(r'\textbf{CMS}', loc='left', fontsize='x-large')
    plt.title(r'63.67 fb$^{-1}$ (13 TeV)', loc='right', fontsize='x-large')
    plt.xlabel(x_title, fontsize='x-large')
    plt.ylabel('mean of '+ varNameMap[target], fontsize='x-large')
    plt.legend(fontsize='large')
    plt.ticklabel_format(style='sci', scilimits=(-2, 3), axis='both', useMathText=True)
    plt.gca().get_yaxis().get_offset_text().set_ha('right')
    plt.gca().get_yaxis().get_offset_text().set_position((0., 1.))
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    fig.savefig('{}/{}_{}_{}_mean.png'.format(plotsdir, EBEE, target, x_var_name))
    fig.savefig('{}/{}_{}_{}_mean.pdf'.format(plotsdir, EBEE, target, x_var_name))
    plt.close(fig)

def weighted_quantiles(a, q, weights=None, **kwargs): 
    if weights is None: 
        return np.quantile(a, q, **kwargs)

    a = np.asarray(a)
    weights = np.asarray(weights)
    idx = np.argsort(a)
    a = a[idx]
    weights = weights[idx]
    quantiles = np.cumsum(weights) / np.sum(weights)
    return np.interp(q, quantiles, a)

def draw_dist_plot(EBEE, df_data, df_mc, qs, x_vars, x_title, x_var_name, target, plotsdir, final=False):

    nq = len(qs)
    var_corr_final = np.array([]).reshape(-1,nq)
    var_corr = np.array([]).reshape(-1,nq)
    var_uncorr = np.array([]).reshape(-1,nq)
    var_data = np.array([]).reshape(-1,nq)
    x_vars_c = np.zeros(len(x_vars)-1)
    for i in range(len(x_vars)-1):
        query_str = x_var_name + ' > ' + str(x_vars[i]) + ' and ' + x_var_name +' < ' + str(x_vars[i+1])
        df_mc_queried = df_mc.query(query_str)

#        var_corr = np.append(var_corr, np.quantile((df_mc.query(query_str))['{}_corr'.format(target)], qs).reshape(-1,nq), axis=0) 
#        var_uncorr = np.append(var_uncorr, np.quantile((df_mc.query(query_str))[target], qs).reshape(-1,nq), axis=0)
#        var_data = np.append(var_data, np.quantile((df_data.query(query_str))[target], qs).reshape(-1,nq), axis=0)

        var_corr = np.append(var_corr, weighted_quantiles(df_mc_queried['{}_corr'.format(target)], qs, weights=df_mc_queried['weight_clf']).reshape(-1,nq), axis=0) 
        var_uncorr = np.append(var_uncorr, weighted_quantiles(df_mc_queried[target], qs, weights=df_mc_queried['weight_clf']).reshape(-1,nq), axis=0)
        var_data = np.append(var_data, weighted_quantiles((df_data.query(query_str))[target], qs).reshape(-1,nq), axis=0)
        if final:
            var_corr_final = np.append(var_corr_final, weighted_quantiles(df_mc_queried['{}_corr_final'.format(target)], qs, weights=df_mc_queried['weight_clf']).reshape(-1,nq), axis=0) 

        x_vars_c[i] = ((x_vars[i] + x_vars[i+1])/2.)

    if EBEE == 'EE' and x_var_name == 'probeScEta': 
        idx_del = np.asarray(np.abs(x_vars_c)<1.5).nonzero()
        var_corr = np.delete(var_corr, idx_del, axis=0)
        var_uncorr = np.delete(var_uncorr, idx_del, axis=0)
        var_data = np.delete(var_data, idx_del, axis=0)
        if final:
            var_corr_final = np.delete(var_corr_final, idx_del, axis=0)

        x_vars_c = np.delete(x_vars_c, idx_del, axis=0)

    var_corr = var_corr.T
    var_uncorr = var_uncorr.T
    var_data = var_data.T
    if final:
        var_corr_final = var_corr_final.T
    
    fig = plt.figure(tight_layout=True)

    if nq > 1: 
        plt.plot(x_vars_c, var_corr[0],   linestyle='--', color='green')
        plt.plot(x_vars_c, var_uncorr[0], linestyle='--', color='red')
        plt.plot(x_vars_c, var_data[0],   linestyle='--', color='blue')

        plt.plot(x_vars_c, var_corr[-1],   linestyle='--', color='green')
        plt.plot(x_vars_c, var_uncorr[-1], linestyle='--', color='red')
        plt.plot(x_vars_c, var_data[-1],   linestyle='--', color='blue')

        if final: 
            plt.plot(x_vars_c, var_corr_final[0], linestyle='--', color='cyan')
            plt.plot(x_vars_c, var_corr_final[-1], linestyle='--', color='cyan')

        for i in range(1,nq//2): 
            plt.fill_between(x_vars_c, var_corr[i],   var_corr[-(i+1)],   color='green', alpha=0.1*(i+1))
            plt.fill_between(x_vars_c, var_uncorr[i], var_uncorr[-(i+1)], color='red',   alpha=0.1*(i+1))
            plt.fill_between(x_vars_c, var_data[i],   var_data[-(i+1)],   color='blue',  alpha=0.1*(i+1))
            if final:
                plt.fill_between(x_vars_c, var_corr_final[i],   var_corr_final[-(i+1)],   color='cyan', alpha=0.1*(i+1))

    if nq%2 != 0: 
        plt.plot(x_vars_c, var_corr[nq//2],   color='green', label='MC corrected')
        plt.plot(x_vars_c, var_uncorr[nq//2], color='red',   label='MC uncorrected')
        plt.plot(x_vars_c, var_data[nq//2],   color='blue',  label='data')
        if final:
            plt.plot(x_vars_c, var_corr_final[nq//2],   color='cyan', label='MC corrected final')
        plt.legend(fontsize='large')

#    plt.title(r'\textbf{CMS}', loc='left', fontsize='x-large')
    plt.title(r'63.67 fb$^{-1}$ (13 TeV)', loc='right', fontsize='x-large')
    plt.xlabel(x_title, fontsize='x-large')
    plt.ylabel(varNameMap[target], fontsize='x-large')
    plt.ticklabel_format(style='sci', scilimits=(-2, 3), axis='both', useMathText=True)
    plt.gca().get_yaxis().get_offset_text().set_ha('right')
    plt.gca().get_yaxis().get_offset_text().set_position((0., 1.))
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    fig.savefig('{}/{}_{}_{}_dist.png'.format(plotsdir, EBEE, target, x_var_name))
    fig.savefig('{}/{}_{}_{}_dist.pdf'.format(plotsdir, EBEE, target, x_var_name))
    plt.close(fig)

def draw_hist(df_data, df_mc, target, fig_name, bins=None, histrange=None, density=False, mc_weights=None, logplot=False, showshift=False, final=False, sysuncer=False):

    if 'EB' in fig_name: 
        EBEE = 'EB'
        etastr = r'$|\eta|<1.4442$'
    elif 'EE' in fig_name: 
        EBEE = 'EE'
        etastr = r'$|\eta|>1.556$'
    else: 
        EBEE = 'all'
        etastr = r''

    nEvt = len(df_data)
    
    mc_weights = mc_weights * nEvt / np.sum(mc_weights)
#    mc_weights = None
    
    fig = plt.figure(tight_layout=True)
    gs = GridSpec(3, 1, figure=fig)
    ax1 = fig.add_subplot(gs[:-1, :])
    ax2 = fig.add_subplot(gs[-1, :])
    

    data_n, bin_edges = np.histogram(df_data[target], range=histrange, bins=bins, density=density)
    data_nerr = np.sqrt(data_n)

    x = np.array([])
    xerr = np.array([])
    for i in range(bins):
        x = np.append(x, (bin_edges[i+1]+bin_edges[i])/2.)
        xerr = np.append(xerr, (bin_edges[i+1]-bin_edges[i])/2.)

    mc_uncorr_n, _, _ = ax1.hist(df_mc[target], range=histrange, bins=bins, density=density, weights=mc_weights, histtype='step', color='red', label='MC uncorrected')

    if showshift:
        mc_shift_n, _, _ = ax1.hist(df_mc['{}_shift'.format(target)], range=histrange, bins=bins, density=density, weights=mc_weights, histtype='step', color='green', label='MC shifted')

    mc_corr_n, _, _ = ax1.hist(df_mc['{}_corr'.format(target)], range=histrange, bins=bins, density=density, weights=mc_weights, histtype='step', color='blue', label='MC corrected')

    if final: 
        mc_final_n, _, _ = ax1.hist(df_mc['{}_corr_final'.format(target)], range=histrange, bins=bins, density=density, weights=mc_weights, histtype='step', color='cyan', label='MC corrected final')
#        mc_final_n, _, _ = ax1.hist(df_mc['{}_corr_final'.format(target)], range=histrange, bins=bins, density=density, weights=mc_weights, histtype='step', color='blue', label='MC corrected final')

    if sysuncer: 
#        mc_corr_n, mc_corr_nsyserr = hists_for_uncer(df_mc, [f'{target}_corr_{i}' for i in range(50)], bins=bins, range=histrange, weights=mc_weights, density=density)
        mc_corr_nerrdn, mc_corr_nerrup = hists_for_uncer2(df_mc, [f'{target}_corr_final_{i}' for i in range(20)], bins=bins, range=histrange, weights=mc_weights, density=density)
        mc_corr_nerrdn = np.append(mc_corr_nerrdn, mc_corr_nerrdn[-1])
        mc_corr_nerrup = np.append(mc_corr_nerrup, mc_corr_nerrup[-1])
#        ax1.step(bin_edges, mc_corr_nsys, where='post', color='blue', label='MC corrected')
        ax1.fill_between(bin_edges, mc_corr_nerrdn, mc_corr_nerrup, step='post', color='blue', alpha=0.2, edgecolor=None, label='syst. uncer.')

    ax1.errorbar(x, data_n, data_nerr, xerr, fmt='.', elinewidth=1., capsize=1., color='black', label='data')
    
    xticks = np.linspace(histrange[0],histrange[1],11)
    ax1.set_xlim(histrange)
    ax1.set_xticks(xticks, labels=[])
    ax1.tick_params(labelsize='large')
    lg = ax1.legend(title=r'$Z \to e^{+}e^{-}$'+'\n'+'Tag and Probe'+'\n'+etastr, frameon=False)
    plt.setp(lg.get_title(), multialignment='center')
#    ax1.set_title(r'\textbf{CMS}', loc='left', fontsize='x-large')
    ax1.set_title(r'63.67 fb$^{-1}$ (13 TeV)', loc='right', fontsize='x-large') # fontsize=16, fontname="Times New Roman"

    binwidth = mathSciNot((histrange[1]-histrange[0])/bins)
    ax1.set_ylabel(f'Events / {binwidth}', fontsize='x-large')
    if logplot: 
        ax1.set_yscale('log')
    else: 
        ax1.ticklabel_format(style='sci', scilimits=(-2, 3), axis='y', useMathText=True)
        ax1.get_yaxis().get_offset_text().set_ha('right')
        ax1.get_yaxis().get_offset_text().set_position((0., 1.))

    mc_uncorr_n = np.where(mc_uncorr_n==0, 1.e-5, mc_uncorr_n)
    mc_corr_n = np.where(mc_corr_n==0, 1.e-5, mc_corr_n)
    ratio_uncorr = data_n / mc_uncorr_n
    ratio_uncorr_err = np.sqrt(data_n + (data_n**2/mc_uncorr_n)) / mc_uncorr_n
    ratio_corr = data_n / mc_corr_n
    ratio_corr_err = np.sqrt(data_n + (data_n**2/mc_corr_n)) / mc_corr_n

    ax2.plot(x, np.ones_like(x), 'k-.')
    ax2.errorbar(x, ratio_uncorr, ratio_uncorr_err, xerr, fmt='.', elinewidth=1., capsize=1., color='red')

    if showshift:
        mc_shift_n = np.where(mc_shift_n==0, 1.e-5, mc_shift_n)
        ratio_shift = data_n / mc_shift_n
        ratio_shift_err = np.sqrt(data_n + (data_n**2/mc_shift_n)) / mc_shift_n
        ax2.errorbar(x, ratio_shift, ratio_shift_err, xerr, fmt='.', elinewidth=1., capsize=1., color='green')

    ax2.errorbar(x, ratio_corr, ratio_corr_err, xerr, fmt='.', elinewidth=1., capsize=1., color='blue')

    if final:
        mc_final_n = np.where(mc_final_n==0, 1.e-5, mc_final_n)
        ratio_final = data_n / mc_final_n
        ratio_final_err = np.sqrt(data_n + (data_n**2/mc_final_n)) / mc_final_n
        ax2.errorbar(x, ratio_final, ratio_final_err, xerr, fmt='.', elinewidth=1., capsize=1., color='cyan')
#        ax2.errorbar(x, ratio_final, ratio_final_err, xerr, fmt='.', elinewidth=1., capsize=1., color='blue')

    if sysuncer: 
        mc_corr_nerrup = np.where(mc_corr_nerrup==0, 1.e-5, mc_corr_nerrup)
        mc_corr_nerrdn = np.where(mc_corr_nerrdn==0, 1.e-5, mc_corr_nerrdn)
        data_nsys = np.append(data_n, data_n[-1])
        ratio_corr_syserrl = data_nsys / mc_corr_nerrup
        ratio_corr_syserrh = data_nsys / mc_corr_nerrdn

        ax2.fill_between(bin_edges, ratio_corr_syserrl, ratio_corr_syserrh, step='post', color='blue', alpha=0.2, edgecolor=None)
    
    ax2.grid(True)
    ax2.set_xlim(histrange)
    ax2.set_ylim(0.8, 1.2)
#    ax2.set_ylim(0.5, 1.5)
    ax2.set_xticks(xticks)
    ax2.set_yticks(np.linspace(0.8, 1.2, 5))
#    ax2.set_yticks(np.linspace(0.5, 1.5, 5))
    ax2.ticklabel_format(style='sci', scilimits=(-2, 3), axis='both', useMathText=True)
    ax2.get_yaxis().get_offset_text().set_ha('right')
    ax2.get_yaxis().get_offset_text().set_position((0., 1.))
    ax2.tick_params(labelsize='large')
    ax2.set_ylabel('Data / MC', fontsize='x-large')
    ax2.set_xlabel(varNameMap[target], fontsize='x-large')
    
    fig.savefig(f'{fig_name}.png')
    fig.savefig(f'{fig_name}.pdf')
    plt.close(fig)

def hists_for_uncer(df, names, bins=100, **kwargs):

    hists = np.array([]).reshape(-1,bins)
    for name in names: 
        h, _ = np.histogram(df[name], bins=bins, **kwargs)
        hists = np.append(hists, h.reshape(1,bins), axis=0)

    return np.mean(hists, axis=0), np.std(hists, axis=0)

def hists_for_uncer2(df, names, bins=100, **kwargs):

    hists = np.array([]).reshape(-1,bins)
    for name in names: 
        h, _ = np.histogram(df[name], bins=bins, **kwargs)
        hists = np.append(hists, h.reshape(1,bins), axis=0)

    return np.min(hists, axis=0), np.max(hists, axis=0)




def main(options):
    variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    isoVarsPh = ['probePhoIso']
    isoVarsCh = ['probeChIso03','probeChIso03worst']
    preshower = ['probeesEnergyOverSCRawEnergy']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    
    EBEE = options.EBEE
#    nEvt = 3500000
    nEvt = options.nEvt

#    df_data = (pd.read_hdf('tmp_dfs/split0.9/df_data_{}_Iso_test.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_data = (pd.read_hdf('dataframes/df_data_{}_Iso_test.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_Iso_test_corr.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_Iso_test_corr_final.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
     
#    df_data = (pd.read_hdf('dataframes/df_data_{}_test.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_test_corr.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_test_corr_final.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_test_corr_final_uncer.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)

    df_data = (pd.read_hdf('tmp_dfs/all/df_data_{}_all.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_all_corr.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_all_corr_final.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_sys/df_mc_{}_all_corr_final.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
    df_mc = (pd.read_hdf('dfs_sys/df_mc_{}_all_corr_final.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_sys/split1/df_mc_{}_all_corr_final.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_sys/split2/df_mc_{}_all_corr_final.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)


#    plotsdir = f'test/plots/split1/{EBEE}'
#    plotsdir = f'test/plots/split2/{EBEE}'
    plotsdir = f'plots/check_correction/{EBEE}'
#    plotsdir = f'plots/check_correction_final/{EBEE}'
#    plotsdir = f'plots/syst_uncer/{EBEE}'

    if EBEE == 'EB': 
        vars_qrnn = variables.copy() 
    else: 
        vars_qrnn = variables+preshower

    vars_corr = ['{}_corr'.format(target) for target in variables] 


    if EBEE != 'EB': 
        vars_corr = vars_corr + ['{}_corr'.format(var) for var in preshower]
    isoVars = isoVarsPh+isoVarsCh
    isoVars_shift = ['{}_shift'.format(var) for var in isoVars]
    isoVars_corr = ['{}_corr'.format(var) for var in isoVars]

    id_start = time()
    weightsEB = 'phoIDmva_weight/HggPhoId_94X_barrel_BDT_v2.weights.xml'
    weightsEE = 'phoIDmva_weight/HggPhoId_94X_endcap_BDT_v2.weights.xml'
    
    phoIDname = 'probePhoIdMVA'
    print('Compute photon ID MVA for data')
    stride = int(df_data.index.size/10) + 1
    df_data[phoIDname] = np.concatenate(Parallel(n_jobs=10, verbose=20)(delayed(helpComputeIdMva)(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars, df_data[ch:ch+stride], 'data', False) for ch in range(0, df_data.index.size, stride))) # variables+isoVars
#    print('Compute photon ID MVA for uncorrected mc')
#    stride = int(df_mc.index.size/10) + 1
#    df_mc[phoIDname] = np.concatenate(Parallel(n_jobs=10, verbose=20)(delayed(helpComputeIdMva)(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars, df_mc[ch:ch+stride], 'data', False) for ch in range(0, df_data.index.size, stride))) # variables+isoVars
###    df_mc[phoIDname] = helpComputeIdMva(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars, df_mc, 'data', False) # +isoVars 
#    print('Compute photon ID MVA for corrected mc')
#    df_mc['{}_corr'.format(phoIDname)] = np.concatenate(Parallel(n_jobs=10, verbose=20)(delayed(helpComputeIdMva)(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars, df_mc[ch:ch+stride], 'qr', False) for ch in range(0, df_data.index.size, stride))) # variables+isoVars
##    df_mc['{}_corr'.format(phoIDname)] = helpComputeIdMva(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars, df_mc, 'qr', False) # +isoVars+preshower 
#    print('Compute photon ID MVA for final correction on mc')
#    df_mc['{}_corr_final'.format(phoIDname)] = np.concatenate(Parallel(n_jobs=10, verbose=20)(delayed(helpComputeIdMva)(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars, df_mc[ch:ch+stride], 'final', False) for ch in range(0, df_data.index.size, stride))) # variables+isoVars
##    df_mc['{}_corr_final'.format(phoIDname)] = helpComputeIdMva(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars, df_mc, 'final', False) # +isoVars+preshower 
    print('time spent in computing photon ID MVA: {}-{:02d}:{:02d}:{:05.2f}'.format(*sec2HMS(time() - id_start)))

    print(df_mc.keys())
        

    # draw plots
    print('Computing weights')
#    df_mc['weight_clf'] = clf_reweight(df_mc, df_data, f'transformer/4d_reweighting_{EBEE}', n_jobs=10)
#    df_mc['weight_clf'] = clf_reweight(df_mc, df_data, f'transformer/4d_reweighting_{EBEE}_Iso', n_jobs=10)
#    df_mc['weight_clf'] = clf_reweight(df_mc, df_data, f'transformer/4d_reweighting_{EBEE}_Iso_0.9', n_jobs=10)
    df_mc['weight_clf'] = clf_reweight(df_mc, df_data, f'transformer/4d_reweighting_{EBEE}_all', n_jobs=10)
#    df_mc['weight_clf'] = 1.

    if EBEE == 'EB':
        histranges = {'probeS4':(0., 1.), 
                      'probeR9':(0., 1.2), 
                      'probeCovarianceIeIp':(-2.e-4, 2.e-4), 
                      'probePhiWidth':(0., 0.2), 
                      'probeSigmaIeIe':(0., 2.e-2), 
                      'probeEtaWidth':(0., 0.05), 
                      'probePhoIso':(0., 6.), 
                      'probeChIso03':(0., 6.), 
                      'probeChIso03worst':(0., 6.)}
    else: 
        histranges = {'probeS4':(0., 1.), 
                      'probeR9':(0., 1.2), 
                      'probeCovarianceIeIp':(-1.5e-3, 1.5e-3), 
                      'probePhiWidth':(0., 0.2), 
                      'probeSigmaIeIe':(0.01, 0.05), 
                      'probeEtaWidth':(0., 0.05), 
                      'probePhoIso':(0., 6.), 
                      'probeChIso03':(0., 6.), 
                      'probeChIso03worst':(0., 6.), 
                      'probeesEnergyOverSCRawEnergy':(0., 0.3)}

    logplots = {'probeS4'            :False, 
                'probeR9'            :False, 
                'probeCovarianceIeIp':False, 
                'probePhiWidth'      :False, 
                'probeSigmaIeIe'     :False, 
                'probeEtaWidth'      :False, 
                'probePhoIso'        :False, 
                'probeChIso03'       :True, 
                'probeChIso03worst'  :False, 
                'probeesEnergyOverSCRawEnergy':False}

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
    xtitles = ['$p_T$ $(\mathrm{GeV})$',   '$\eta$',     '$\\rho$ $(\mathrm{GeV})$', '$\phi$'  ]
    xnames  = ['probePt', 'probeScEta', 'rho',     'probePhi']
    qs = np.array([0.025, 0.16, 0.5, 0.84, 0.975])

    for target in isoVars: #variables+preshower+isoVars 

        if EBEE == 'EB' and target in preshower: 
            continue

        fig_name = '{}/data_mc_dist_{}_{}'.format(plotsdir, EBEE, target)
#        fig_name = '{}/data_mc_dist_{}_{}_uncer'.format(plotsdir, EBEE, target)
    
        if target in preshower: 
            query_preshower = 'probeScEta<-1.653 or probeScEta>1.653'
            draw_hist(df_data.query(query_preshower), df_mc.query(query_preshower), target, fig_name, bins, histranges[target], mc_weights=(df_mc.query(query_preshower))['weight_clf'], logplot=logplots[target], final=False)
             
            for x, xtitle, xname in zip(xs, xtitles, xnames): 
                draw_mean_plot(EBEE, df_data.query(query_preshower), df_mc.query(query_preshower), x, xtitle, xname, target, plotsdir, final=False)
                draw_dist_plot(EBEE, df_data.query(query_preshower), df_mc.query(query_preshower), qs, x, xtitle, xname, target, plotsdir, final=False)
        elif target in isoVars: 
            draw_hist(df_data, df_mc, target, fig_name, bins, histranges[target], mc_weights=df_mc['weight_clf'], logplot=logplots[target], showshift=True, final=False)
             
            if EBEE == 'EB':
                query_iso = f'probeSigmaIeIe<0.0105'
            else: 
                query_iso = f'probeSigmaIeIe<0.028'
            for x, xtitle, xname in zip(xs, xtitles, xnames): 
                draw_mean_plot(EBEE, df_data.query(query_iso), df_mc.query(query_iso), x, xtitle, xname, target, plotsdir, final=False)
                draw_dist_plot(EBEE, df_data.query(query_iso), df_mc.query(query_iso), qs, x, xtitle, xname, target, plotsdir, final=False)
        else: 
            draw_hist(df_data, df_mc, target, fig_name, bins, histranges[target], mc_weights=df_mc['weight_clf'], logplot=logplots[target], final=False, sysuncer=False)
             
            for x, xtitle, xname in zip(xs, xtitles, xnames): 
                draw_mean_plot(EBEE, df_data, df_mc, x, xtitle, xname, target, plotsdir, final=True)
                draw_dist_plot(EBEE, df_data, df_mc, qs, x, xtitle, xname, target, plotsdir, final=True)

  
    draw_hist(
        df_data, df_mc, 
        phoIDname, 
        '{}/data_mc_dist_{}_{}'.format(plotsdir, EBEE, phoIDname),
#        '{}/data_mc_dist_{}_{}_uncer'.format(plotsdir, EBEE, phoIDname),
#        '{}/data_mc_dist_{}_{}_uncer2'.format(plotsdir, EBEE, phoIDname),
        bins = bins, 
        histrange = (-0.8, 1.),
        mc_weights = df_mc['weight_clf'], 
        final = False, 
#        sysuncer = True, 
        )

    for x, xtitle, xname in zip(xs, xtitles, xnames): 
        draw_mean_plot(EBEE, df_data, df_mc, x, xtitle, xname, phoIDname, plotsdir, final=False)
        draw_dist_plot(EBEE, df_data, df_mc, qs, x, xtitle, xname, phoIDname, plotsdir, final=False)




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-n','--nEvt', action='store', type=int, required=True)
    options = parser.parse_args()
    main(options)
