import os
import argparse
import uproot
import pandas as pd


def make_dataframe(path, tree, data_key, EBEE, dfDir, dfname, cut=None, split=None, sys=False):

    Branches = ['probeScEta', 'weight', 'probeSigmaRR', 'tagChIso03', 'tagR9', 'tagPhiWidth_Sc', 'probePt', 'tagSigmaRR', 'puweight', 'tagEleMatch', 'tagPhi', 'probeScEnergy', 'nvtx',  
                'tagPhoIso', 'run', 'tagScEta', 'probeEleMatch', 'tagPt', 'rho', 'tagS4', 'tagSigmaIeIe', 'tagCovarianceIpIp', 'tagCovarianceIeIp', 'tagScEnergy', 'tagChIso03worst', 
                'probePhi', 'mass', 'probeCovarianceIpIp', 'tagEtaWidth_Sc', 'probeHoE', 'probeFull5x5_e1x5', 'probeFull5x5_e5x5', 'probeNeutIso', 'probePassEleVeto']

    variables = ['probeEtaWidth_Sc', 'probeR9', 'probeS4', 'probePhiWidth_Sc', 'probeSigmaIeIe', 'probeCovarianceIeIp', 
                'probeChIso03', 'probeChIso03worst', 'probeesEnergyOverSCRawEnergy'] 
 
    rename_dict = {'tagPhiWidth_Sc': 'tagPhiWidth', 
                   'tagEtaWidth_Sc': 'tagEtaWidth'}
    if data_key == 'data':
        branches = Branches + variables + ['probePhoIso03'] 
        rename_dict.update({'probeEtaWidth_Sc': 'probeEtaWidth', 'probePhiWidth_Sc': 'probePhiWidth', 'probePhoIso03':'probePhoIso'})
    elif data_key == 'mc': 
        vars_raw = [(var[:var.find('_')] if '_' in var else var) for var in variables+['probePhoIso']]
        branches = Branches + ['{}_uncorr'.format(var) for var in vars_raw]
        rename_dict.update({'{}_uncorr'.format(var):var for var in vars_raw})
   
    ptmin = 25.
    ptmax = 150.
    etamin = -2.5
    etamax = 2.5
    phimin = -3.14
    phimax = 3.14
    
    
    print(f'load root files from {path}, tree name: {tree}')
    root_file = uproot.open(path)
    up_tree = root_file[tree]
    
    df = up_tree.arrays(branches, library='pd')
    print(df.keys())
    print('renaming data frame columns: ', rename_dict)
    df.rename(columns=rename_dict, inplace=True)
    print(df.keys())
    print(df)
    
    df.query('probePt>@ptmin and probePt<@ptmax and probeScEta>@etamin and probeScEta<@etamax and probePhi>@phimin and probePhi<@phimax',inplace=True)
    
    if EBEE == 'EB': 
        df.query('probeScEta>-1.4442 and probeScEta<1.4442',inplace=True)
    elif EBEE == 'EE': 
        df.query('probeScEta<-1.556 or probeScEta>1.556',inplace=True)
    
    if cut is not None: 
        print('apply additional cut: ', cut)
        df.query(cut,inplace=True)
    
    df = df.sample(frac=1.).reset_index(drop=True)
    
    if sys: 
        df_train1 = df[0:int(0.45*df.index.size)]
        df_train2 = df[int(0.45*df.index.size):int(0.9*df.index.size)]
        df_test = df[int(0.9*df.index.size):]
        df_train1.to_hdf('{}/{}_train_split1.h5'.format(dfDir,dfname),'df',mode='w',format='t')
        df_train2.to_hdf('{}/{}_train_split2.h5'.format(dfDir,dfname),'df',mode='w',format='t')
        df_test.to_hdf('{}/{}_test.h5'.format(dfDir,dfname),'df',mode='w',format='t')
        print('{}/{}_(train/test).h5 have been created'.format(dfDir,dfname))
    else:
        if split is not None: 
            df_train = df[0:int(split*df.index.size)]
            df_test = df[int(split*df.index.size):]
            df_train.to_hdf('{}/{}_train.h5'.format(dfDir,dfname),'df',mode='w',format='t')
            df_test.to_hdf('{}/{}_test.h5'.format(dfDir,dfname),'df',mode='w',format='t')
            print('{}/{}_(train/test).h5 have been created'.format(dfDir,dfname))
        else: 
            df.to_hdf('{}/{}.h5'.format(dfDir,dfname),'df',mode='w',format='t')
            print('{}/{}.h5 have been created'.format(dfDir,dfname))


def main(options):

    path = {'data': './root_files/outputData.root', 
              'mc': './root_files/outputMC.root'}
    tree = {'data': 'tagAndProbeDumper/Data_13TeV_All', 
              'mc': 'tagAndProbeDumper/DYJetsToLL_amcatnloFXFX_13TeV_All'}
    
    cut = 'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeChIso03<6 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0'
    
    cutIso = {'EB': 'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeSigmaIeIe<0.0105 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0',
              'EE': 'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeSigmaIeIe<0.028 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0'}

    cutplots = 'tagPt>40 and probePt>20 and mass>80 and mass<100 and probePassEleVeto==0 and tagScEta<2.5 and tagScEta>-2.5' 
    
    data_key = options.data_key
    EBEE = options.EBEE 
    year = 2018
    split = 0.9
#    dfDir = f'./tmp_dfs/split{split}'
    dfDir = f'./tmp_dfs/sys'

    if not os.path.exists(dfDir): 
        os.makedirs(dfDir)

#    make_dataframe(path[data_key], tree[data_key], data_key, EBEE, dfDir, 'df_{}_all'.format(data_key, EBEE))

#    make_dataframe(path[data_key], tree[data_key], data_key, EBEE, dfDir, 'df_{}_{}'.format(data_key, EBEE), cut, split)
#    make_dataframe(path[data_key], tree[data_key], data_key, EBEE, dfDir, 'df_{}_{}_Iso'.format(data_key, EBEE), cutIso[EBEE], split)


    make_dataframe(path[data_key], tree[data_key], data_key, EBEE, dfDir, 'df_{}_{}'.format(data_key, EBEE), cut, sys=True)
    make_dataframe(path[data_key], tree[data_key], data_key, EBEE, dfDir, 'df_{}_{}_Iso'.format(data_key, EBEE), cutIso[EBEE], sys=True)

#    make_dataframe(path[data_key], tree[data_key], data_key, EBEE, dfDir, 'df_{}_{}_all'.format(data_key, EBEE), cutplots)



if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-d','--data_key', action='store', type=str, required=True)
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    options = parser.parse_args()
    main(options)
