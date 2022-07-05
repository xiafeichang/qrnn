from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as sta
from ..plotting.plot_dmc_hist import plot_dmc_hist
import xgboost as xgb
import sys
import os

class utils(object):

    @classmethod
    def get_quantile(cls,df,df_ref,var,var_ref,weights=None,inv=False):
        if weights is None:
            cdf = np.vstack((np.hstack((np.array(range(len(df_ref.index)))/float(df_ref.index)),[1])),np.hstack((np.sort(df_ref[var_ref].values),1.2*np.sort(df_ref[var_ref].values)[-1])))
        else:
            sort_df = df_ref.sort_values(var_ref)
            w_cum = np.cumsum(sort_df[weights].values,dtype=np.float64)
            # print(w_cum,np.divide(w_cum,w_cum[-1]))
            cdf = np.vstack((np.hstack((np.divide(w_cum,w_cum[-1]),[1],[999999])),np.hstack((np.sort(df_ref[var_ref].values),[999998],[999999]))))

        if inv:
            return np.apply_along_axis(cls.transform_inv,0,df[var].values,cdf)
        else:
            return np.apply_along_axis(cls.transform,0,df[var].values,cdf)

    @staticmethod
    def transform(val,cdf):
        ind = np.searchsorted(cdf[1],val)
        if val.any()>(cdf[1][-1]):
            print(val, ind, cdf[0][ind])# cdf[1][ind]
        return cdf[0][ind]

    @staticmethod
    def transform_inv(val,cdf):
        ind = np.searchsorted(cdf[0],val)
        if val.any()>(cdf[1][-1]):
            print(val, ind, cdf[1][ind])# cdf[1][ind]
        return cdf[1][ind]

    @staticmethod
    def para(x, a, b, c):
        return a*x**2 + b*x + c

    @staticmethod
    def poly3(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d

    @staticmethod
    def poly4(x, a, b, c, d, e):
        return a*x**4 + b*x**3 + c*x**2 + d*x + e

    @staticmethod
    def const(x, a):
        return a*np.ones(x.shape[0])

    @staticmethod
    def findBand(mat, bins, weights, cutoff=0.):

        shifted_hists = []
        for i in range(mat.shape[1]):
            hist,_ = np.histogram(mat[:,i],bins=bins,weights=weights)
            shifted_hists.append(hist)

        mini = []
        maxi = []
        for k in range(shifted_hists[0].shape[0]):
            bin_occ = np.array([shifted_hists[i][k] for i in range(mat.shape[1])])
            mini.append(bin_occ.min())
            maxi.append(bin_occ.max())
        maxi = np.array(maxi)
        mini = np.array(mini)

        if cutoff is not None:
            nonz = mini[mini > cutoff]
            for idx, mi in enumerate(mini):
                if mi < cutoff and idx < 0.5 * bins.shape[0]:
                    mini[idx] = nonz[0]
                elif mi < cutoff and idx > 0.5 * bins.shape[0]:
                    mini[idx] = nonz[-1]

        return mini, maxi

    @staticmethod
    def clf_reweight(df_mc,df_data,n_jobs=1,cut=None):
        clf = xgb.XGBClassifier(learning_rate=0.05,n_estimators=500,max_depth=10,gamma=0,n_jobs=n_jobs)
        features = ['probePt','rho','probeScEta','probePhi']
        if cut is not None:
            X_data = df_data.query(cut, engine='python').sample(min(min(df_mc.query(cut, engine='python').index.size,df_data.query(cut, engine='python').index.size), 1000000)).loc[:,features].values
            X_mc = df_mc.query(cut, engine='python').sample(min(min(df_mc.query(cut, engine='python').index.size,df_data.query(cut, engine='python').index.size), 1000000)).loc[:,features].values
        else:
            X_data = df_data.sample(min(min(df_mc.index.size,df_data.index.size), 1000000)).loc[:,features].values
            X_mc = df_mc.sample(min(min(df_mc.index.size,df_data.index.size), 1000000)).loc[:,features].values
        X = np.vstack([X_data,X_mc])
        y = np.vstack([np.ones((X_data.shape[0],1)),np.zeros((X_mc.shape[0],1))])
        X, y = shuffle(X,y)

        clf.fit(X,y)
        eps = 1.e-3
        return np.apply_along_axis(lambda x: x[1]/(x[0]+eps), 1, clf.predict_proba(df_mc.loc[:,features].values))

class systShift(object):

    def __init__(self, df1, df2, shiftFctn=None):

        self.df = pd.DataFrame(columns=['diffTrainings', 'diffTrainings_transformed', 'newPhoIDcorrAll1', 'newPhoID1', 'newPhoIDtrcorrAll1', 'newPhoIDtr1', 'probePt', 'probeScEta', 'rho', 'weight_clf'], data=np.vstack((df1['newPhoIDcorrAll'] - df2['newPhoIDcorrAll'],df1['newPhoIDtrcorrAll'] - df2['newPhoIDtrcorrAll'],df1['newPhoIDcorrAll'],df1['newPhoID'],df1['newPhoIDtrcorrAll'],df1['newPhoIDtr'],df1['probePt'],df1['probeScEta'],df1['rho'],df1['weight_clf'])).T)
        if shiftFctn is None:
            self.const = True
            self.shiftFctn = utils.const
        else:
            self.const = False
            self.shiftFctn = shiftFctn

    def getShiftPars(self, correctEdges=False):

        bins = np.linspace(0,1,101)
        self.df['newPhoIDtrcorrAll_bin'] = pd.cut(self.df['newPhoIDtrcorrAll1'],bins=bins,labels=np.arange(bins.shape[0]-1))
        groupby = self.df.groupby(self.df['newPhoIDtrcorrAll_bin'])
        self.xc = 0.5*(bins[1:] + bins[:-1])
        cutoff = self.df['diffTrainings_transformed'].quantile(q=0.95)
        self.stand = np.array([groupby.get_group(i).loc[abs(groupby.get_group(i)['diffTrainings_transformed'])<cutoff,'diffTrainings_transformed'].std() for i in range(100)])
        if correctEdges:
            indMax = self.stand == self.stand.max()
            diffMax = np.abs(self.xc - self.xc[indMax] * np.ones_like(self.xc))
            self.stand = (1. - diffMax) * self.stand + diffMax * self.stand.max() * np.ones_like(self.stand)
        if self.const:
            self.shift_pars = [self.df['diffTrainings_transformed'].quantile(q=0.84)]
        else:
            self.shift_pars, pcov_std = opt.curve_fit(self.shiftFctn, self.xc, self.stand)

    def getShift(self, val):

        return self.shiftFctn(val,*self.shift_pars)

    def plotFit(self, saveDir=None, label=None):

        plt.figure()
        plt.errorbar(self.xc,self.stand,xerr=np.ones(100)*(self.xc[1]-self.xc[0])*0.5,marker='.',ls='None')
        plt.grid()
        plt.xlabel('newPhoIDtr')
        plt.ylabel('Std(difference 2 trainings)')
        plt.plot(self.xc,self.getShift(self.xc))
        plt.ylim(0,0.1)
        if saveDir is not None:
            if label is not None:
                plt.savefig('{}/plot_syst_diffStdFit_{}.png'.format(saveDir,label))
                plt.savefig('{}/plot_syst_diffStdFit_{}.pdf'.format(saveDir,label))
            else:
                plt.savefig('{}/plot_syst_diffStdFit.png'.format(saveDir))
                plt.savefig('{}/plot_syst_diffStdFit.pdf'.format(saveDir))

class systematics(object):

    def __init__(self, df, shifts, shiftFctn, nomVar='probePhoIdMVAtr'):

        self.df = df
        self.shifts = np.array(shifts)
        self.shiftFctn = shiftFctn
        self.nomVar = nomVar

    def multiShift(self, val, stat_corr):

        std = self.shiftFctn(val)/stat_corr
        shift_mat = np.outer(std,self.shifts)
        val_mat = np.outer(val,np.ones(self.shifts.shape[0]))
        return val_mat + shift_mat

    def applShifts(self, stat_corr):

        args = [stat_corr]
        shifted = np.apply_along_axis(self.multiShift,0,self.df['probePhoIdMVAtr'].values, *args).T
        for i in range(self.shifts.shape[0]):
            self.df['probePhoIdMVAtr_shift{}'.format(i)] = shifted[i]

    def getBand(self, bins, weights, cutoff):

        self.bins = np.array(bins)

        mat = self.df.loc[:,['probePhoIdMVAtr_shift{}'.format(i) for i in range(self.shifts.shape[0])]].values
        self.mini, self.maxi = utils.findBand(mat, bins, weights, cutoff)

    def plotBand(self, df_data, saveDir=None, cut=None, label=None, zoom=False):

        dic = {}
        dic['var'] = 'probePhoIdMVAtr'
        dic['bins'] = self.bins.shape[0]-1
        dic['xmin'] = self.bins.min()
        dic['xmax'] = self.bins.max()
        dic['weightstr_mc'] = 'weight_clf'
        dic['ratio_lim'] = (0.5,1.5) if not zoom else (0.9,1.1)
        dic['type'] = 'dataMC'
        cutBool = False
        if label is None:
            label = 'IdMVA syst'
        if cut is not None:
            dic['cut'] = cut
            cutBool = True
            label = label + ' EB' if 'abs(probeScEta)<1.4442' in cut else label + ' EE'

        # df_data['probePhoIdMVAtr'] = df_data['newPhoIDtr']
        plltt = plot_dmc_hist(self.df, df_data=df_data, ratio=True, norm=True, cut_str='', label=label, **dic)
        plltt.draw()

        xc = 0.5*(plltt.bins[1:]+plltt.bins[:-1])
        norm_fac = plltt.data.shape[0] / plltt.mc_weights_cache.sum()
        # print norm_fac
        plltt.fig.axes[0].fill_between(xc,norm_fac*self.maxi,norm_fac*self.mini,color='purple',alpha=0.3,label='Systematic uncertainity',step='mid')
        rdatamc_syst_down = np.divide(plltt.data_hist, norm_fac*self.mini, dtype=np.float)
        rdatamc_syst_up = np.divide(plltt.data_hist, norm_fac*self.maxi, dtype=np.float)
        rdatamc_syst_down[np.isinf(rdatamc_syst_down)] = 999999
        plltt.fig.axes[1].fill_between(xc,rdatamc_syst_down,rdatamc_syst_up,color='purple',alpha=0.3,label='Systematic uncertainity',step='mid')
        legend = plltt.fig.axes[0].legend()
        if cutBool:
            figsize = plltt.fig.get_size_inches()*plltt.fig.dpi
            pos = legend.get_window_extent()
            ann_pos, lr, tb = plltt.get_annot_pos(pos, figsize)
            lc = {'left': 'left', 'right': 'right'}
            plltt.get_tex_cut()
            plltt.fig.axes[0].annotate(r'\begin{{{0}}}{1}\end{{{0}}}'.format('flush{}'.format(lr), plltt.cut_str_tex), tuple(ann_pos), fontsize=14, xycoords=plltt.fig.axes[0].get_legend(), bbox={'boxstyle': 'square', 'alpha': 0, 'fc': 'w', 'pad': 0}, ha=lr, va=tb)

        if saveDir is not None:
            plltt.save(saveDir)

    def saveSystFile(self, ofile, df_data):

        for i in [0,19]:
                self.df['probePhoIdMVA_shift{}'.format(i)] = utils.get_quantile(self.df,df_data,'probePhoIdMVAtr_shift{}'.format(i),'probePhoIdMVA',weights='weight',inv=True)

        self.df['probePhoIdMVA_bin'],bins = pd.qcut(self.df.loc[self.df['probePhoIdMVA_shift19']<999,'probePhoIdMVA'],q=1000,labels=np.arange(1000),retbins=True)
        groupby = self.df.groupby(self.df['probePhoIdMVA_bin'])
        x = 0.5*(bins[1:]+bins[:-1])
        x = np.append(x, [1,9999])
        x = np.insert(x, 0, [-9999,-1])

        y_down = np.array([groupby.get_group(i)['probePhoIdMVA_shift0'].median() for i in np.arange(1000)])
        y_up = np.array([groupby.get_group(i)['probePhoIdMVA_shift19'].median() for i in np.arange(1000)])

        y_down = np.append(y_down, [1,9999])
        y_down = np.insert(y_down, 0, [-9999,-1])
        y_up = np.append(y_up, [1,9999])
        y_up = np.insert(y_up, 0, [-9999,-1])

        # This needs to be changed because when running for EE none of the two conditions seem to be
        # fulfilled
        #if np.all(np.abs(self.df['probeScEta'].values) > 1.56):
            #title = 'trasfhee'
        #elif np.all(np.abs(self.df['probeScEta'].values) < 1.4442):
            #title = 'trasfheb'

        if np.all(np.abs(self.df['probeScEta'].values) < 1.4442):
            title = 'trasfheb'
        else:
            title = 'trasfhee'


        if os.path.exists(ofile):
            out_df = pd.read_hdf(ofile)
            if title == 'trasfheb':
                out_df['x_eb'] = x
            elif title == 'trasfhee':
                out_df['x_ee'] = x
            out_df['{}down'.format(title)] = y_down
            out_df['{}up'.format(title)] = y_up
        else:
            columns = ['x_eb','{}down'.format(title),'{}up'.format(title)] if title == 'trasfheb' else ['x_ee','{}down'.format(title),'{}up'.format(title)]
            out_df = pd.DataFrame(data=np.vstack((x,y_down,y_up)).T, columns=columns)

        out_df.to_hdf(ofile,'df',mode='w',format='t')
