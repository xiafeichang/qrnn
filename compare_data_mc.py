import time
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import xgboost as xgb
from sklearn.utils import shuffle
from time import time
import pickle

from qrnn import trainQuantile, predict, scale
from mylib.Corrector import Corrector, applyCorrection
from mylib.transformer import transform, inverse_transform


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

def draw_hist(df_data, df_mc, target, fig_name, bins=None, histrange=None, density=False, mc_weights=None):
    fig = plt.figure(tight_layout=True)
    gs = GridSpec(3, 1, figure=fig)
    ax1 = fig.add_subplot(gs[:-1, :])
    ax2 = fig.add_subplot(gs[-1, :])
    
    #mc_uncorr_n, _, _ = ax1.hist(inverse_transform(df_mc[target], transformer_file, target), range=histrange, bins=nbin, density=True, histtype='step', color='red', label='MC uncorrected')
    #mc_corr_n, _, _ = ax1.hist(inverse_transform(df_mc['{}_corr'.format(target)], transformer_file, target), range=histrange, bins=nbin, density=True, histtype='step', color='blue', label='MC corrected')
    #data_n, bin_edges = np.histogram(inverse_transform(df_data[target], transformer_file, target), range=histrange, bins=nbin, density=True)
    
    mc_uncorr_n, _, _ = ax1.hist(df_mc[target], range=histrange, bins=bins, density=density, weights=mc_weights, histtype='step', color='red', label='MC uncorrected')
    mc_corr_n, _, _ = ax1.hist(df_mc['{}_corr'.format(target)], range=histrange, bins=bins, density=density, weights=mc_weights, histtype='step', color='blue', label='MC corrected')
    data_n, bin_edges = np.histogram(df_data[target], range=histrange, bins=bins, density=density)
    
    x = np.array([])
    xerr = np.array([])
    for i in range(len(data_n)):
        x = np.append(x, (bin_edges[i+1]+bin_edges[i])/2.)
        xerr = np.append(xerr, (bin_edges[i+1]-bin_edges[i])/2.)
    data_nerr = np.sqrt(data_n*xerr*2./nEvnt)
    ax1.errorbar(x, data_n, data_nerr, xerr, fmt='.', elinewidth=1., capsize=1., color='black', label='data')
    
    xticks = np.linspace(histrange[0],histrange[1],10)
    ax1.set_xlim(histrange)
    ax1.set_xticks(xticks, labels=[])
    ax1.ticklabel_format(style='sci', scilimits=(-2, 3), axis='y')
    ax1.legend()
    ax1.set_title(target)
    
#    ratio_uncorr = data_n / mc_uncorr_n
#    ratio_uncorr_err = np.sqrt((data_n*xerr*2./nEvnt) + (data_n**2/mc_uncorr_n)*(xerr*2./nEvnt)) / mc_uncorr_n
#    ratio_corr = data_n / mc_corr_n
#    ratio_corr_err = np.sqrt((data_n*xerr*2./nEvnt) + (data_n**2/mc_corr_n)*(xerr*2./nEvnt)) / mc_corr_n

    ratio_uncorr = (mc_uncorr_n - data_n) / data_n
    ratio_uncorr_err = np.sqrt(((mc_uncorr_n+data_n)*xerr*2./nEvnt) + ((mc_uncorr_n-data_n)**2/data_n)*(xerr*2./nEvnt)) / data_n
    ratio_corr = (mc_corr_n - data_n) / data_n
    ratio_corr_err = np.sqrt(((mc_corr_n+data_n)*xerr*2./nEvnt) + ((mc_corr_n-data_n)**2/data_n)*(xerr*2./nEvnt)) / data_n
    
    ax2.plot(x, np.zeros_like(x), 'k-.')
    ax2.errorbar(x, ratio_uncorr, ratio_uncorr_err, xerr, fmt='.', elinewidth=1., capsize=1., color='red')
    ax2.errorbar(x, ratio_corr, ratio_corr_err, xerr, fmt='.', elinewidth=1., capsize=1., color='blue')
    
    ax2.set_xlim(histrange)
#    ax2.set_ylim(0.85, 1.15)
    ax2.set_ylim(-0.15, 0.15)
    ax2.set_xticks(xticks)
    ax2.ticklabel_format(style='sci', scilimits=(-2, 3), axis='both')
    
    fig.savefig(fig_name)
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

print('Computing weights')
df_mc['weight_clf'] = clf_reweight(df_mc, df_data, 'transformer/4d_reweighting_EB', n_jobs=10)

#histranges = []

# correct
#target = variables[5]
#for target, histrange in zip(variables, histranges): 
for target in variables: 
    features = kinrho# + ['{}_corr'.format(x) for x in variables[:variables.index(target)]]
    
    X = df_mc.loc[:,features]
    Y = df_mc.loc[:,target]
    
    models_mc = 'models/{}_{}_{}'.format('mc', EBEE, target)
    models_d = 'models/{}_{}_{}'.format('data', EBEE, target)
    df_mc['{}_corr'.format(target)] = applyCorrection(models_mc, models_d, X, Y, diz=False)
    print(df_mc)
     
    histrange = (-4., 4.)
    bins = 75
    fig_name = 'plots/check_correction/data_mc_dist_{}_{}'.format(EBEE, target)

    draw_hist(df_data, df_mc, target, fig_name, bins, histrange, density=True, mc_weights=df_mc['weight_clf'])
     
