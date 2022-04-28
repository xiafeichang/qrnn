import pickle
import gzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from mylib.transformer import transform, inverse_transform


def load_clf(clf_name): 
    clf = pickle.load(gzip.open(clf_name))
    return clf['clf']

def draw_plot(df_data, df_mc, x_vars, x_title, x_var_name, target, fig_name):

    peak_true_data = np.zeros(len(x_vars)-1)
    tail_true_data = np.zeros(len(x_vars)-1)
    peak_pred_data = np.zeros(len(x_vars)-1)
    tail_pred_data = np.zeros(len(x_vars)-1)

    peak_true_mc = np.zeros(len(x_vars)-1)
    tail_true_mc = np.zeros(len(x_vars)-1)
    peak_pred_mc = np.zeros(len(x_vars)-1)
    tail_pred_mc = np.zeros(len(x_vars)-1)

    x_vars_c = np.zeros(len(x_vars)-1)
    for i in range(len(x_vars)-1):
        query_str = x_var_name + ' > ' + str(x_vars[i]) + ' and ' + x_var_name +' <= ' + str(x_vars[i+1])
        df_data_cut = df_data.query(query_str)
        df_mc_cut = df_mc.query(query_str)

        nTot_data = len(df_data_cut)
        nTot_mc = len(df_mc_cut)

        peak_true_data[i] = len(df_data_cut.query(f'p0t_{target} == 0'))/nTot_data 
        tail_true_data[i] = len(df_data_cut.query(f'p0t_{target} != 0'))/nTot_data 
#        peak_pred_data[i] = np.average(df_data_cut[f'p0t_{target}_pp'])
#        tail_pred_data[i] = np.average(df_data_cut[f'p0t_{target}_pt'])

        peak_true_mc[i] = len(df_mc_cut.query(f'p0t_{target}_shift == 0'))/nTot_mc 
        tail_true_mc[i] = len(df_mc_cut.query(f'p0t_{target}_shift != 0'))/nTot_mc 
#        peak_pred_mc[i] = np.average(df_mc_cut[f'p0t_{target}_pp'])
#        tail_pred_mc[i] = np.average(df_mc_cut[f'p0t_{target}_pt'])

        x_vars_c[i] = ((x_vars[i] + x_vars[i+1])/2.)

    fig = plt.figure(tight_layout=True)

    plt.plot(x_vars_c, peak_true_data, 'r--', label='data peak')
    plt.plot(x_vars_c, tail_true_data, 'r-.', label='data tail')
    plt.plot(x_vars_c, peak_true_mc, 'b--', label='mc shifted peak')
    plt.plot(x_vars_c, tail_true_mc, 'b-.', label='mc shifted tail')

#    plt.plot(x_vars_c, peak_true_data, 'r--', label='data peak true')
#    plt.plot(x_vars_c, tail_true_data, 'r-.', label='data tail true')
#    plt.plot(x_vars_c, peak_pred_data, 'ro', fillstyle='none', label='data peak pred')
#    plt.plot(x_vars_c, tail_pred_data, 'rv', fillstyle='none', label='data tail pred')

#    plt.plot(x_vars_c, peak_true_mc, 'b--', label='mc peak true')
#    plt.plot(x_vars_c, tail_true_mc, 'b-.', label='mc tail true')
#    plt.plot(x_vars_c, peak_pred_mc, 'bo', fillstyle='none', label='mc peak pred')
#    plt.plot(x_vars_c, tail_pred_mc, 'bv', fillstyle='none', label='mc tail pred')

    plt.xlabel(x_title)
    plt.ylim(0.,1.)
    plt.legend(ncol=2)
    fig.savefig(f'{fig_name}_{x_var_name}.png')
    plt.close(fig)


def main(): 
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    isoVarsPh = ['probePhoIso']
    isoVarsCh = ['probeChIso03','probeChIso03worst']

    EBEE = 'EB'
    nEvt = 3500000
    df_data = (pd.read_hdf(f'dataframes/df_data_{EBEE}_Iso_test.h5')).sample(nEvt, random_state=100).reset_index(drop=True)
    df_mc = (pd.read_hdf(f'test/dfs_corr/df_mc_{EBEE}_Iso_test_corr.h5')).sample(nEvt, random_state=100).reset_index(drop=True)

#    transformer_file = 'data_{}'.format(EBEE)
#    df_data.loc[:,kinrho] = transform(df_data.loc[:,kinrho], transformer_file, kinrho)
#    df_mc.loc[:,kinrho] = transform(df_mc.loc[:,kinrho], transformer_file, kinrho)

    features = kinrho
    target = isoVarsPh[0]
    df_data[f'p0t_{target}'] = np.apply_along_axis(lambda x: 0 if x==0 else 1, 0, df_data[target].values.reshape(1,-1))
    df_mc[f'p0t_{target}_shift'] = np.apply_along_axis(lambda x: 0 if x==0 else 1, 0, df_mc[f'{target}_shift'].values.reshape(1,-1))
    
#    clf_data = load_clf(f'chained_models/data_{EBEE}_clf_{target}.pkl')
#    proba_data_clf = clf_data.predict_proba(df_data.loc[:,features])
#    df_data[f'p0t_{target}_pp'] = proba_data_clf[:, 0]
#    df_data[f'p0t_{target}_pt'] = proba_data_clf[:, 1]
#
#    clf_mc = load_clf(f'chained_models/mc_{EBEE}_clf_{target}.pkl')
#    proba_mc_clf = clf_mc.predict_proba(df_mc.loc[:,features])
#    df_mc[f'p0t_{target}_pp'] = proba_mc_clf[:, 0]
#    df_mc[f'p0t_{target}_pt'] = proba_mc_clf[:, 1]
#
#    df_data.loc[:,kinrho] = inverse_transform(df_data.loc[:,kinrho], transformer_file, kinrho)
#    df_mc.loc[:,kinrho] = inverse_transform(df_mc.loc[:,kinrho], transformer_file, kinrho)

    pTs = np.arange(25., 55., 1.5)
    etas = np.arange(-1.45, 1.45, 0.15)
    rhos = np.arange(0., 50., 2.)
    phis = np.arange(-3.15, 3.15, 0.3)

    plotsdir = 'plots/other_test'
    draw_plot(df_data, df_mc, pTs,  '$p_T$',   'probePt',    target, f'{plotsdir}/check_shift_{EBEE}_{target}')
    draw_plot(df_data, df_mc, etas, '$\eta$',  'probeScEta', target, f'{plotsdir}/check_shift_{EBEE}_{target}')
    draw_plot(df_data, df_mc, rhos, '$\\rho$', 'rho',        target, f'{plotsdir}/check_shift_{EBEE}_{target}')
    draw_plot(df_data, df_mc, phis, '$\phi$',  'probePhi',   target, f'{plotsdir}/check_shift_{EBEE}_{target}')
 
if __name__ == '__main__': 
    main()
