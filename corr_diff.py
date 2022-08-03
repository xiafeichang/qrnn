import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('cairo')
from matplotlib import pyplot as plt
import seaborn as sns

from check_results import clf_reweight


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize='x-large')

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize='large')
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize='large')

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, minann=0., **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is None:
        threshold = np.abs(data).max()/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if abs(data[i,j]) > minann: 
                kw.update(color=textcolors[int(abs(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

    return texts

def draw_corr_plots(corr, title, figname, cmap='seismic', cbarlabel='', minann=None, **kwargs): 

    xticklabels = [(var.replace('probe','')).replace('esEnergyOverSCRawEnergy','EesOverEsc') for var in corr.columns]
    yticklabels = [(var.replace('probe','')).replace('esEnergyOverSCRawEnergy','EesOverEsc') for var in corr.index]
    fig, ax = plt.subplots(figsize=(10,8), tight_layout=True)
#    sns.heatmap(corr, vmin=vmin, vmax=vmax, linewidth=linewidth, center=center, annot=annot, cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels, ax=ax)
#    plt.title(title)
#    plt.yticks(rotation='0', ha='right')
#    plt.xticks(rotation='30', ha='right')

    im, cbar = heatmap(corr.values, yticklabels, xticklabels, ax=ax, cmap=cmap, cbarlabel=cbarlabel, **kwargs)
    texts = annotate_heatmap(im, minann=minann, threshold=0.5*max(kwargs['vmax'], kwargs['vmin']))

    ax.set_title(title, fontsize='x-large')

    ax.hlines([3.5, 9.5],3.5,9.5, colors='blue', linewidth=2.5)
    ax.vlines([3.5, 9.5],3.5,9.5, colors='blue', linewidth=2.5)

    ax.hlines([9.5, 12.5],9.5,12.5, colors='purple', linewidth=2.5)
    ax.vlines([9.5, 12.5],9.5,12.5, colors='purple', linewidth=2.5)

    fig.savefig(f'{figname}.png')
    fig.savefig(f'{figname}.pdf')

def _wcorr(arr1,arr2,weights):
    m1 = np.average(arr1,weights=weights)*np.ones_like(arr1)
    m2 = np.average(arr2,weights=weights)*np.ones_like(arr2)
    cov_11 = float((weights*(arr1-m1)**2).sum()/weights.sum())
    cov_22 = float((weights*(arr2-m2)**2).sum()/weights.sum())
    cov_12 = float((weights*(arr1-m1)*(arr2-m2)).sum()/weights.sum())
    return cov_12/np.sqrt(cov_11*cov_22)

def weighted_corr(df, weights):
    return pd.DataFrame([[_wcorr(df[var1], df[var2], weights) for var1 in df.columns] for var2 in df.columns], columns=df.columns, index=df.columns)



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
    df_mc = pd.read_hdf('dfs_corr/df_mc_{}_all_corr_final.h5'.format(EBEE))

#    plotsdir = f'plots/check_correction/{EBEE}'
    plotsdir = f'plots/check_correction_final/{EBEE}'

#    df_mc['weight_clf'] = clf_reweight(df_mc, df_data, f'transformer/4d_reweighting_{EBEE}', n_jobs=10)
#    df_mc['weight_clf'] = clf_reweight(df_mc, df_data, f'transformer/4d_reweighting_{EBEE}_Iso', n_jobs=10)
#    df_mc['weight_clf'] = clf_reweight(df_mc, df_data, f'transformer/4d_reweighting_{EBEE}_Iso_0.9', n_jobs=10)
    df_mc['weight_clf'] = clf_reweight(df_mc, df_data, f'transformer/4d_reweighting_{EBEE}_all', n_jobs=10)
#    df_mc['weight_clf'] = 1.

    print(df_mc)


    varss = variables+isoVarsPh+isoVarsCh
    if EBEE == 'EE': 
        varss = varss + preshower

    varss_corr = [f'{var}_corr_final' for var in varss]
    rename_dict = {f'{var}_corr_final':var for var in varss}

#    vars_corr = [f'{var}_corr_final' for var in variables]
#    isoVars = isoVarsPh+isoVarsCh
#    isoVars_corr = ['{}_corr_final'.format(var) for var in isoVars]
#
#    rename_dict = {f'{var}_corr_final':var for var in variables+isoVars}


    pTbin = ['probePt<35', 'probePt>35 and probePt<50', 'probePt<50']
    pTbinstr = [r'$p_{T} < 35$', r'$35 < p_{T} < 50$', r'$p_{T} > 50$']
    for i in range(len(pTbin)):
        cut = pTbin[i]
        df_data_ = df_data.query(cut)
        df_mc_ = df_mc.query(cut)

#        corr_data = df_data.loc[:, kinrho+variables+isoVars].corr()
#        corr_mc = df_mc.loc[:, kinrho+variables+isoVars].corr()
#        corr_mc_corr = df_mc.loc[:, kinrho+vars_corr+isoVars_corr].corr()

        corr_data = df_data_.loc[:, kinrho+varss].corr()
        corr_mc = weighted_corr(df_mc_.loc[:, kinrho+varss], weights=df_mc_['weight_clf'])
        corr_mc_corr = weighted_corr(df_mc_.loc[:, kinrho+varss_corr], weights=df_mc_['weight_clf'])

        corr_mc_corr.rename(index=rename_dict, columns=rename_dict, inplace=True)

        vrange = (-15., 15.) 
        draw_corr_plots(
            100*corr_data, 
            'Correlation in data', 
            f'{plotsdir}/correlation_{EBEE}_all_pTbin{i}_data', 
            cbarlabel = r'$Corr_{data}$ (\%)', 
            minann = 10., 
            vmin=-100., vmax=100., 
            )

        draw_corr_plots(
            100*(corr_data-corr_mc), 
            'Correlation difference with '+pTbinstr[i], 
            f'{plotsdir}/correlation_{EBEE}_all_pTbin{i}_uncorr', 
            cbarlabel = r'($Corr_{data} - Corr_{mc}$) (\%)', 
            minann = 1., 
            vmin=vrange[0], vmax=vrange[1], 
            )

        draw_corr_plots(
            100*(corr_data-corr_mc_corr), 
            'Correlation difference with '+pTbinstr[i], 
            f'{plotsdir}/correlation_{EBEE}_all_pTbin{i}_corr', 
            cbarlabel = r'($Corr_{data} - Corr_{mc_{corr}}$) (\%)', 
            minann = 1., 
            vmin=vrange[0], vmax=vrange[1], 
            )



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    options = parser.parse_args()
    main(options)
