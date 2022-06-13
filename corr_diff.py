import argparse
import pandas as pd
import matplotlib
matplotlib.use('cairo')
from matplotlib import pyplot as plt
import seaborn as sns


def draw_corr_plots(corr, title, figname, vmin=None, vmax=None, xticklabels=None, yticklabels=None, xrotation=30, cmap='seismic', center=0., annot=True, linewidth=0.5): 

    if xticklabels is None: 
        xticklabels = [var.replace('probe','') for var in corr.columns]
    if yticklabels is None: 
        yticklabels = [var.replace('probe','') for var in corr.index]
    fig, ax = plt.subplots(figsize=(10,8), tight_layout=True)
    sns.heatmap(corr, vmin=vmin, vmax=vmax, linewidth=linewidth, center=center, annot=annot, cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels, ax=ax)
    plt.title(title)
    plt.yticks(rotation='0', ha='right')
    plt.xticks(rotation='30', ha='right')
    fig.savefig(f'{figname}.png')
    fig.savefig(f'{figname}.pdf')




def main(options): 
    variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    isoVarsPh = ['probePhoIso']
    isoVarsCh = ['probeChIso03','probeChIso03worst']
    preshower = ['probeesEnergyOverSCRawEnergy']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    
    EBEE = options.EBEE

#    df_data = pd.read_hdf('dataframes/df_data_{}_Iso_test.h5'.format(EBEE))
#    df_mc = pd.read_hdf('dfs_corr/df_mc_{}_Iso_test_corr.h5'.format(EBEE))
#    df_mc = pd.read_hdf('dfs_corr/df_mc_{}_Iso_test_corr_final.h5'.format(EBEE))
     
#    df_data = pd.read_hdf('dataframes/df_data_{}_test.h5'.format(EBEE))
#    df_mc = pd.read_hdf('dfs_corr/df_mc_{}_test_corr.h5'.format(EBEE))
#    df_mc = pd.read_hdf('dfs_corr/df_mc_{}_test_corr_final.h5'.format(EBEE))
#    df_mc = pd.read_hdf('dfs_corr/df_mc_{}_test_corr_final_uncer.h5'.format(EBEE))

    df_data = pd.read_hdf('tmp_dfs/all/df_data_{}_all.h5'.format(EBEE))
    df_mc = pd.read_hdf('dfs_corr/df_mc_{}_all_corr.h5'.format(EBEE))

    cut = 'probePt>35 and probePt<50'
    df_data.query(cut, inplace=True)
    df_mc.query(cut, inplace=True)

    plotsdir = f'plots/check_correction/{EBEE}'
#    plotsdir = f'plots/check_correction_final/{EBEE}'

    vars_corr = [f'{var}_corr' for var in variables]
    isoVars = isoVarsPh+isoVarsCh
    isoVars_shift = ['{}_shift'.format(var) for var in isoVars]
    isoVars_corr = ['{}_corr'.format(var) for var in isoVars]


    corr_data = df_data.loc[:, kinrho+variables+isoVars].corr()
    corr_mc = df_mc.loc[:, kinrho+variables+isoVars].corr()
    corr_mc_corr = df_mc.loc[:, kinrho+vars_corr+isoVars_corr].corr()

    rename_dict = {f'{var}_corr':var for var in variables+isoVars}
    corr_mc_corr.rename(index=rename_dict, columns=rename_dict, inplace=True)

    vrange = (-15., 15.) 
    draw_corr_plots(
        100*(corr_data-corr_mc), 
        r'Correlation difference with $35 < p_{T} < 50$ ($Corr_{data} - Corr_{mc}$) (%)', 
        f'{plotsdir}/correlation_{EBEE}_all', 
        vrange[0], vrange[1], 
        )

    draw_corr_plots(
        100*(corr_data-corr_mc_corr), 
        r'Correlation difference with $35 < p_{T} < 50$ ($Corr_{data} - Corr_{mc_{corr}}$) (%)', 
        f'{plotsdir}/correlation_{EBEE}_all_corr', 
        vrange[0], vrange[1], 
        )



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    options = parser.parse_args()
    main(options)
