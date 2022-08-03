import argparse
import pandas as pd
import numpy as np
import scipy.optimize as opt
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('cairo')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'Helvetica'

from mylib.IdMVAComputer import helpComputeIdMva
from mylib.tools import weighted_quantiles



def check_phoID(df, dtype, EBEE):

    variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth','probePhoIso','probeChIso03','probeChIso03worst']
    if EBEE == 'EE':
        variables = variables + ['probeesEnergyOverSCRawEnergy']

    weightsEB = 'phoIDmva_weight/HggPhoId_94X_barrel_BDT_v2.weights.xml'
    weightsEE = 'phoIDmva_weight/HggPhoId_94X_endcap_BDT_v2.weights.xml'

    stride = int(df.index.size/10) + 1
    if dtype != 'split': 
        if 'probePhoIdMVA' not in df.keys(): 
            print(f'photon ID not in df, Compute photon ID MVA for {dtype}')
            df['probePhoIdMVA'] = np.concatenate(Parallel(n_jobs=10, verbose=20)(delayed(helpComputeIdMva)(weightsEB, weightsEE, EBEE, variables, df[ch:ch+stride], 'data', False) for ch in range(0, df.index.size, stride))) 
    if 'probePhoIdMVA_corr' not in df.keys(): 
        print(f'corrected photon ID not in df, Compute corrected photon ID MVA for {dtype}')
        df['probePhoIdMVA_corr'] = np.concatenate(Parallel(n_jobs=10, verbose=20)(delayed(helpComputeIdMva)(weightsEB, weightsEE, EBEE, variables, df[ch:ch+stride], 'qr', False) for ch in range(0, df.index.size, stride))) 
    if 'probePhoIdMVA_corr_final' not in df.keys(): 
        print(f'finally corrected photon ID not in df, Compute finally corrected photon ID MVA for {dtype}')
        df['probePhoIdMVA_corr_final'] = np.concatenate(Parallel(n_jobs=10, verbose=20)(delayed(helpComputeIdMva)(weightsEB, weightsEE, EBEE, variables, df[ch:ch+stride], 'final', False) for ch in range(0, df.index.size, stride))) 



def train_quantile_transformer(df_ref, var_ref, weights=None):
    if weights is None:
        return np.vstack((np.hstack((np.array(range(len(df_ref.index)))/float(df_ref.index)),[1])),np.hstack((np.sort(df_ref[var_ref].values),1.2*np.sort(df_ref[var_ref].values)[-1])))
    else:
        sort_df = df_ref.sort_values(var_ref)
        w_cum = np.cumsum(sort_df[weights].values,dtype=np.float64)
        return np.vstack((np.hstack((np.divide(w_cum,w_cum[-1]),[1],[999999])),np.hstack((np.sort(df_ref[var_ref].values),[999998],[999999]))))

def transform(a, transformer, inv=False):
    # to transform the photon ID into a flat diatribution in (0., 1.)
    if inv:
        return np.apply_along_axis(_transform_inv, 0, a, transformer)
    else:
        return np.apply_along_axis(_transform, 0, a, transformer)

def _transform(val, cdf):
    ind = np.searchsorted(cdf[1],val)
#    if val.any()>(cdf[1][-1]):
#        print(val, ind, cdf[0][ind])# cdf[1][ind]
    return cdf[0][ind]

def _transform_inv(val, cdf):
    ind = np.searchsorted(cdf[0],val)
#    if val.any()>(cdf[1][-1]):
#        print(val, ind, cdf[1][ind])# cdf[1][ind]
    return cdf[1][ind]

def para(x, a, b, c):
    return a*x**2 + b*x + c

def poly3(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def poly4(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def const(x, a):
    return a*np.ones(x.shape[0])



def bin_flat(df, var, nbin=100):
    bins = np.linspace(0., 1., nbin+1)
    df[f'{var}_flatbin'] = pd.cut(df[f'{var}_flat'], bins=bins, labels=np.arange(nbin))
    return (bins[1:] + bins[:-1]) / 2.

def get_sigmas(df, var, method='quantile', fitfunc=None, xs=None, figname='phoIdErr'):

    if var.endswith('_flat'):
        gbvar = f'{var}bin'
    else: 
        gbvar = f'{var}_flatbin'

    gb = df[f'{var}_sysdiff'].groupby(df[gbvar])

    dn = np.array([g.quantile(0.16) for _, g in gb])
    up = np.array([g.quantile(0.84) for _, g in gb])
    
    fig, ax = plt.subplots(tight_layout=True)
    ax.errorbar(xs, dn, xerr=0.5*(xs[1]-xs[0]), fmt='.', elinewidth=1., capsize=1., color='blue', label=r'$1\sigma_{-}$')
    ax.errorbar(xs, up, xerr=0.5*(xs[1]-xs[0]), fmt='.', elinewidth=1., capsize=1., color='black', label=r'$1\sigma_{+}$')

    if fitfunc is None:  
        df[f'{var}_sysdn'] = gb.transform(lambda x: x.quantile(0.16))
        df[f'{var}_sysup'] = gb.transform(lambda x: x.quantile(0.84))
    else: 
        dn_curve_par, _ = opt.curve_fit(fitfunc, xs, dn)
        up_curve_par, _ = opt.curve_fit(fitfunc, xs, up)
        df[f'{var}_sysdn'] = fitfunc(df[var], *dn_curve_par)
        df[f'{var}_sysup'] = fitfunc(df[var], *up_curve_par)

        ax.plot(xs, fitfunc(xs, *dn_curve_par), color='red', label=(r'${0:.2f}x^{{2}}+{1:.2f}x+{2:.2f}$'.format(*dn_curve_par)).replace('+-', '-'))
        ax.plot(xs, fitfunc(xs, *up_curve_par), color='green', label=(r'${0:.2f}x^{{2}}+{1:.2f}x+{2:.2f}$'.format(*up_curve_par)).replace('+-', '-'))

    ax.legend()
    ax.set_xlim(0.,1.)
    ax.set_xlabel('flattened photon ID MVA score', fontsize='x-large')
    ax.set_ylabel('error in flattened score', fontsize='x-large')
    fig.savefig(f'{figname}.png')
    fig.savefig(f'{figname}.pdf')

def sys_shift(df, var, nshifts=5):
    shifts = np.linspace(df[var]+df[f'{var}_sysdn'], df[var]+df[f'{var}_sysup'], nshifts, endpoint=True)
    for i in range(nshifts):
        df[f'{var}_{i}'] = shifts[i]





def main(options):

    EBEE = options.EBEE
#    df = pd.read_hdf(f'dfs_corr/df_mc_{EBEE}_all_corr_final.h5') 
    df = pd.read_hdf(f'dfs_sys/df_mc_{EBEE}_all_corr_final.h5') 
    df1 = pd.read_hdf(f'dfs_sys/split1/df_mc_{EBEE}_all_corr_final.h5') 
    df2 = pd.read_hdf(f'dfs_sys/split2/df_mc_{EBEE}_all_corr_final.h5') 

    check_phoID(df, 'mc', EBEE)
    check_phoID(df1, 'split', EBEE)
    check_phoID(df2, 'split', EBEE)

    var = 'probePhoIdMVA_corr_final'
    transformer = train_quantile_transformer(df, var, weights='weight')

    df[f'{var}_flat'] = transform(df[var].values, transformer) 
    df1[f'{var}_flat'] = transform(df1[var].values, transformer) 
    df2[f'{var}_flat'] = transform(df2[var].values, transformer) 

    df[f'{var}_flat_sysdiff'] = df1[f'{var}_flat'] - df2[f'{var}_flat']

    xs_fb = bin_flat(df, var, 100)
    print(df.keys())
    print(df1.keys())
    print(df2.keys())

    get_sigmas(df, f'{var}_flat', fitfunc=para, xs=xs_fb, figname=f'plots/syst_uncer/phoIdErr_{EBEE}')

    nshifts = 20
    sys_shift(df, f'{var}_flat', nshifts)
    print([k for k in df.keys()])

    for i in range(nshifts): 
        df[f'{var}_{i}'] = transform(df[f'{var}_flat_{i}'], transformer, inv=True)

    df.to_hdf(f'dfs_sys/df_mc_{EBEE}_all_corr_final.h5', 'df', mode='w', format='t')




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
#    requiredArgs.add_argument('-n','--nEvt', action='store', type=int, required=True)
    options = parser.parse_args()
    main(options)
