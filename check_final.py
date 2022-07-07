import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('cairo')
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import xgboost as xgb
from sklearn.utils import shuffle
from time import time
import pickle
import gzip

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
    var_data_mean = np.zeros(len(x_vars)-1)
    x_vars_c = np.zeros(len(x_vars)-1)
    for i in range(len(x_vars)-1):
        query_str = x_var_name + ' > ' + str(x_vars[i]) + ' and ' + x_var_name +' < ' + str(x_vars[i+1])

        var_corr_mean[i] = np.mean(np.mean((df_mc.query(query_str)).loc[:,[f'{target}_corr_{j}' for j in range(20)]], axis=1) 
                                   - (df_mc.query(query_str))[target]) 
        var_data_mean[i] = np.mean((df_data.query(query_str))[f'{target}_corr'] - (df_data.query(query_str))[target])

        x_vars_c[i] = ((x_vars[i] + x_vars[i+1])/2.)

    if EBEE == 'EE' and x_var_name == 'probeScEta': 
        idx_del = np.asarray(np.abs(x_vars_c)<1.5).nonzero()
        var_corr_mean = np.delete(var_corr_mean, idx_del, axis=0)
        var_data_mean = np.delete(var_data_mean, idx_del, axis=0)

        x_vars_c = np.delete(x_vars_c, idx_del, axis=0)

    fig = plt.figure(tight_layout=True)
    plt.plot(x_vars_c, var_corr_mean,   color='red', label='predicted')
    plt.plot(x_vars_c, var_data_mean,   color='blue',  label='true')

    plt.title(r'$\bf{CMS}$ $\it{Work\ in\ Progress}$', loc='left')
    plt.title(f'UL 2018 {EBEE}', loc='right')
    plt.gca().yaxis.get_offset_text().set_x(-0.1)
    plt.xlabel(x_title)
    plt.ylabel('mean of {} correction'.format(target))
    plt.legend()
    fig.savefig('{}/{}_{}_{}_mean_corr.png'.format(plotsdir, EBEE, target, x_var_name))
    fig.savefig('{}/{}_{}_{}_mean_corr.pdf'.format(plotsdir, EBEE, target, x_var_name))
    plt.close(fig)

def draw_dist_plot(EBEE, df_data, df_mc, qs, x_vars, x_title, x_var_name, target, plotsdir):

    nq = len(qs)
    var_corr = np.array([]).reshape(-1,nq)
    var_data = np.array([]).reshape(-1,nq)
    x_vars_c = np.zeros(len(x_vars)-1)
    for i in range(len(x_vars)-1):
        query_str = x_var_name + ' > ' + str(x_vars[i]) + ' and ' + x_var_name +' < ' + str(x_vars[i+1])

        var_corr = np.append(var_corr, np.quantile(np.mean((df_mc.query(query_str)).loc[:,[f'{target}_corr_{j}' for j in range(20)]], axis=1)
                                                    - (df_mc.query(query_str))[target], 
                                                   qs).reshape(-1,nq), axis=0) 
        var_data = np.append(var_data, np.quantile((df_data.query(query_str))[f'{target}_corr'] - (df_data.query(query_str))[target], 
                                                   qs).reshape(-1,nq), axis=0)

        x_vars_c[i] = ((x_vars[i] + x_vars[i+1])/2.)

    if EBEE == 'EE' and x_var_name == 'probeScEta': 
        idx_del = np.asarray(np.abs(x_vars_c)<1.5).nonzero()
        var_corr = np.delete(var_corr, idx_del, axis=0)
        var_data = np.delete(var_data, idx_del, axis=0)

        x_vars_c = np.delete(x_vars_c, idx_del, axis=0)

    var_corr = var_corr.T
    var_data = var_data.T
    
    fig = plt.figure(tight_layout=True)

    if nq > 1: 
        plt.plot(x_vars_c, var_corr[0],   linestyle='--', color='red')
        plt.plot(x_vars_c, var_data[0],   linestyle='--', color='blue')

        plt.plot(x_vars_c, var_corr[-1],   linestyle='--', color='red')
        plt.plot(x_vars_c, var_data[-1],   linestyle='--', color='blue')

        for i in range(1,nq//2): 
            plt.fill_between(x_vars_c, var_corr[i],   var_corr[-(i+1)],   color='red', alpha=0.1*(i+1))
            plt.fill_between(x_vars_c, var_data[i],   var_data[-(i+1)],   color='blue',  alpha=0.1*(i+1))

    if nq%2 != 0: 
        plt.plot(x_vars_c, var_corr[nq//2],   color='red', label='predicted')
        plt.plot(x_vars_c, var_data[nq//2],   color='blue',  label='true')
        plt.legend()

    plt.title(r'$\bf{CMS}$ $\it{Work\ in\ Progress}$', loc='left')
    plt.title(f'UL 2018 {EBEE}', loc='right')
    plt.xlabel(x_title)
    plt.ylabel(f'{target} correction')
    fig.savefig('{}/{}_{}_{}_dist_corr.png'.format(plotsdir, EBEE, target, x_var_name))
    fig.savefig('{}/{}_{}_{}_dist_corr.pdf'.format(plotsdir, EBEE, target, x_var_name))
    plt.close(fig)

def draw_dist_std_plot(EBEE, df_mc, qs, x_vars, x_title, x_var_name, target, plotsdir):

    nq = len(qs)
    var_corr = np.array([]).reshape(-1,nq)
    x_vars_c = np.zeros(len(x_vars)-1)
    for i in range(len(x_vars)-1):
        query_str = x_var_name + ' > ' + str(x_vars[i]) + ' and ' + x_var_name +' < ' + str(x_vars[i+1])

        var_corr = np.append(var_corr, np.quantile(np.std((df_mc.query(query_str)).loc[:,[f'{target}_corr_{j}' for j in range(20)]], axis=1), 
                                                   qs).reshape(-1,nq), axis=0) 

        x_vars_c[i] = ((x_vars[i] + x_vars[i+1])/2.)

    if EBEE == 'EE' and x_var_name == 'probeScEta': 
        idx_del = np.asarray(np.abs(x_vars_c)<1.5).nonzero()
        var_corr = np.delete(var_corr, idx_del, axis=0)

        x_vars_c = np.delete(x_vars_c, idx_del, axis=0)

    var_corr = var_corr.T
    
    fig = plt.figure(tight_layout=True)

    if nq > 1: 
        plt.plot(x_vars_c, var_corr[0],   linestyle='--', color='blue')

        plt.plot(x_vars_c, var_corr[-1],   linestyle='--', color='blue')

        for i in range(1,nq//2): 
            plt.fill_between(x_vars_c, var_corr[i],   var_corr[-(i+1)],   color='blue', alpha=0.1*(i+1))

    if nq%2 != 0: 
        plt.plot(x_vars_c, var_corr[nq//2],   color='blue')

    plt.title(r'$\bf{CMS}$ $\it{Work\ in\ Progress}$', loc='left')
    plt.title(f'UL 2018 {EBEE}', loc='right')
    plt.xlabel(x_title)
    plt.ylabel(f'{target}_corr std')
    fig.savefig('{}/{}_{}_{}_dist_std.png'.format(plotsdir, EBEE, target, x_var_name))
    fig.savefig('{}/{}_{}_{}_dist_std.pdf'.format(plotsdir, EBEE, target, x_var_name))
    plt.close(fig)

def draw_hist(df_data, df_mc, nEvt, target, fig_name, bins=None, histrange=None, density=False, mc_weights=None, logplot=False, sysuncer=False):

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
    
    mc_uncorr_n, bin_edges, _ = ax1.hist(df_mc[target], range=histrange, bins=bins, density=density, weights=mc_weights, histtype='step', color='red', label='MC uncorrected')

    x = np.array([])
    xerr = np.array([])
    for i in range(bins):
        x = np.append(x, (bin_edges[i+1]+bin_edges[i])/2.)
        xerr = np.append(xerr, (bin_edges[i+1]-bin_edges[i])/2.)

    if sysuncer: 
        mc_corr_n, mc_corr_nsyserr = hists_for_uncer(df_mc, [f'{target}_corr_{i}' for i in range(20)], bins=bins, range=histrange, weights=mc_weights, density=density)
        mc_corr_nsys = np.append(mc_corr_n, mc_corr_n[-1])
        mc_corr_nsyserr = np.append(mc_corr_nsyserr, mc_corr_nsyserr[-1])
        ax1.step(bin_edges, mc_corr_nsys, where='post', color='blue', label='MC corrected')
        ax1.fill_between(bin_edges, mc_corr_nsys-mc_corr_nsyserr, mc_corr_nsys+mc_corr_nsyserr, step='post', color='blue', alpha=0.3, label='sys. err.')
    else:  
        mc_corr_n, _, _ = ax1.hist(df_mc['{}_corr'.format(target)], range=histrange, bins=bins, density=density, weights=mc_weights, histtype='step', color='blue', label='MC corrected')

    data_n, _ = np.histogram(df_data[target], range=histrange, bins=bins, density=density)
    data_nerr = np.sqrt(data_n)
    ax1.errorbar(x, data_n, data_nerr, xerr, fmt='.', elinewidth=1., capsize=1., color='black', label='data')
    
    xticks = np.linspace(histrange[0],histrange[1],10)
    ax1.set_xlim(histrange)
    ax1.set_xticks(xticks, labels=[])
#    ax1.ticklabel_format(style='sci', scilimits=(-2, 3), axis='y')
    ax1.legend()
    ax1.set_title(r'$\bf{CMS}$ $\it{Work\ in\ Progress}$', loc='left')
    ax1.set_title(r'UL 2018 $\bf{'+str(EBEE)+'}$', loc='right') # fontsize=16, fontname="Times New Roman"
    ax1.set_ylabel(f'Events / {(histrange[1]-histrange[0])/bins}')
    if logplot: 
        ax1.set_yscale('log')
    
    ratio_uncorr = data_n / mc_uncorr_n
    ratio_uncorr_err = np.sqrt(data_n + (data_n**2/mc_uncorr_n)) / mc_uncorr_n
    ratio_corr = data_n / mc_corr_n
    ratio_corr_err = np.sqrt(data_n + (data_n**2/mc_corr_n)) / mc_corr_n

    ax2.plot(x, np.ones_like(x), 'k-.')
    ax2.errorbar(x, ratio_uncorr, ratio_uncorr_err, xerr, fmt='.', elinewidth=1., capsize=1., color='red')
    ax2.errorbar(x, ratio_corr, ratio_corr_err, xerr, fmt='.', elinewidth=1., capsize=1., color='blue')

    if sysuncer: 
        data_nsys = np.append(data_n, data_n[-1])
        ratio_corr_syserrl = data_nsys / (mc_corr_nsys + mc_corr_nsyserr)
        ratio_corr_syserrh = data_nsys / (mc_corr_nsys - mc_corr_nsyserr)

        ax2.fill_between(bin_edges, ratio_corr_syserrl, ratio_corr_syserrh, step='post', color='blue', alpha=0.3)
    
    ax2.grid(True)
    ax2.set_xlim(histrange)
    ax2.set_ylim(0.8, 1.2)
    ax2.set_xticks(xticks)
    ax2.ticklabel_format(style='sci', scilimits=(-2, 3), axis='both')
    ax2.set_ylabel('Data / MC')
    ax2.set_xlabel(target)
    
    fig.savefig(f'{fig_name}.png')
    fig.savefig(f'{fig_name}.pdf')
    plt.close(fig)

def hists_for_uncer(df, names, bins=100, **kwargs):

    hists = np.array([]).reshape(-1,bins)
    for name in names: 
        h, _ = np.histogram(df[name], bins=bins, **kwargs)
        hists = np.append(hists, h.reshape(1,bins), axis=0)

    return np.mean(hists, axis=0), np.std(hists, axis=0)




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
#    df_mc = pd.read_hdf('dfs_corr/df_mc_{}_Iso_test_corr.h5'.format(EBEE))
     
#    df_data = (pd.read_hdf('dataframes/df_data_{}_test.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_data = (pd.read_hdf('dfs_corr/df_mc_{}_test_corr.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
    df_data = (pd.read_hdf('dfs_corr/df_mc_{}_test_corr_final.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_test_corr_final.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)
    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_test_corr_final_uncer.h5'.format(EBEE))).sample(nEvt, random_state=100).reset_index(drop=True)


#    plotsdir = f'plots/check_correction/{EBEE}'
    plotsdir = f'plots/check_correction_final/predAndtrue'

    isoVars = isoVarsPh+isoVarsCh
       

    # draw plots
    print('Computing weights')
    df_mc['weight_clf'] = clf_reweight(df_mc, df_data, f'transformer/4d_reweighting_{EBEE}', n_jobs=10)
#    df_mc['weight_clf'] = clf_reweight(df_mc, df_data, f'transformer/4d_reweighting_{EBEE}_Iso', n_jobs=10)

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
                      'probeChIso03worst':(0., 10.), 
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
    xtitles = ['$p_T$',   '$\eta$',     '$\\rho$', '$\phi$'  ]
    xnames  = ['probePt', 'probeScEta', 'rho',     'probePhi']
    qs = np.array([0.025, 0.16, 0.5, 0.84, 0.975])

    for target in variables: #isoVars+variables+preshower 
        fig_name = '{}/data_mc_dist_{}_{}_uncer'.format(plotsdir, EBEE, target)
    
        if target in preshower: 
            query_preshower = 'probeScEta<-1.653 or probeScEta>1.653'
#            draw_hist(df_data.query(query_preshower), df_mc.query(query_preshower), nEvt, target, fig_name, bins, histranges[target], mc_weights=df_mc['weight_clf'], logplot=logplots[target])
             
            for x, xtitle, xname in zip(xs, xtitles, xnames): 
                draw_mean_plot(EBEE, df_data.query(query_preshower), df_mc.query(query_preshower), x, xtitle, xname, target, plotsdir)
                draw_dist_plot(EBEE, df_data.query(query_preshower), df_mc.query(query_preshower), qs, x, xtitle, xname, target, plotsdir)
#        elif target in isoVars: 
##            draw_hist(df_data, df_mc, nEvt, target, fig_name, bins, histranges[target], mc_weights=df_mc['weight_clf'], logplot=logplots[target])
#             
#            query_iso = f'{target}!=0'
#            for x, xtitle, xname in zip(xs, xtitles, xnames): 
#                draw_mean_plot(EBEE, df_data.query(query_iso), df_mc.query(query_iso), x, xtitle, xname, target, plotsdir)
#                draw_dist_plot(EBEE, df_data.query(query_iso), df_mc.query(query_iso), qs, x, xtitle, xname, target, plotsdir)
        else: 
#            draw_hist(df_data, df_mc, nEvt, target, fig_name, bins, histranges[target], mc_weights=df_mc['weight_clf'], logplot=logplots[target], sysuncer=True)
             
            for x, xtitle, xname in zip(xs, xtitles, xnames): 
                draw_mean_plot(EBEE, df_data, df_mc, x, xtitle, xname, target, plotsdir)
                draw_dist_plot(EBEE, df_data, df_mc, qs, x, xtitle, xname, target, plotsdir)
                draw_dist_std_plot(EBEE, df_mc, qs, x, xtitle, xname, target, plotsdir)

  



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-n','--nEvt', action='store', type=int, required=True)
    options = parser.parse_args()
    main(options)
