import warnings 
import argparse
import time
import pandas as pd

from nn import predict
from mylib.transformer import transform, inverse_transform
from mylib.tools import *

def main(options): 
    variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    isoVarsPh = ['probePhoIso']
    isoVarsCh = ['probeChIso03','probeChIso03worst']
    preshower = ['probeesEnergyOverSCRawEnergy']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    
    EBEE = options.EBEE
    data_type = options.data_type
    spl = options.split
    var_type = options.var_type
    
    if spl in [1, 2]: 
        if var_type == 'all': 
#            iptdir = f'dfs_sys/split{spl}'
            iptdir = f'dfs_sys'
            inputmc = f'df_mc_{EBEE}_all_corr_final.h5'
        else:
            iptdir = f'dfs_sys/split{spl}'
            inputmc = f'df_mc_{EBEE}_Iso_{data_type}_split{spl}_corr.h5'
    else: 
        if var_type == 'all': 
#            iptdir = f'dfs_corr'
            iptdir = f'dfs_sys'
            inputmc = f'df_mc_{EBEE}_all_corr_final.h5'
        else:
            iptdir = 'dfs_corr'
            inputmc = f'df_mc_{EBEE}_Iso_{data_type}_corr.h5'
        warnings.warn(f"Wrong argument '-s' ('--split'), argument must have value 1 or 2. Now using defalt dataframe {iptdir}/{inputmc}")
    if data_type == 'test': 
        inputmc = inputmc.replace(f'_split{spl}','')
       
    print(f'loading dataframe from {iptdir}/{inputmc}')
    df_mc = (pd.read_hdf(f'{iptdir}/{inputmc}')).reset_index(drop=True)

#    df_mc = (pd.read_hdf('weighted_dfs/df_mc_{}_Iso_{}.h5'.format(EBEE,data_type))).reset_index(drop=True)
#    df_mc = (pd.read_hdf(f'dfs_corr/df_mc_{EBEE}_all_corr.h5')).reset_index(drop=True)
#    df_mc = (pd.read_hdf(f'dfs_corr/df_mc_{EBEE}_all_corr_final.h5')).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_Iso_{}_corr.h5'.format(EBEE,data_type))).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_Iso_{}_corr_final.h5'.format(EBEE,data_type))).reset_index(drop=True)

    if spl in [1,2]:
        modeldir = f'models/split{spl}'
        outdir = f'dfs_sys/split{spl}'
    else:
        modeldir = 'chained_models'
#        outdir   = 'dfs_corr'
        outdir   = 'dfs_sys'
    print(f'using models from {modeldir}, corrected dataframes will be saved in {outdir}')
 
    if 'probeScEta_orignal' not in df_mc.keys():
        df_mc['probeScEta_orignal'] = df_mc['probeScEta']
    transformer_file = 'data_{}'.format(EBEE)
    df_mc.loc[:,kinrho] = transform(df_mc.loc[:,kinrho], transformer_file, kinrho)

    trans_file_corr_diff = f'mc_{EBEE}'
    for target in isoVarsPh+isoVarsCh: 

        if target in isoVarsPh: 
            features = kinrho + isoVarsPh
        else: 
            features = kinrho + isoVarsCh
    
        model_file = '{}/mc_{}_{}_final'.format(modeldir, EBEE, target)
    
        target_name = f'{target}_corr_diff'
        df_mc.loc[df_mc[f'{target}_shift']==0, f'{target}_corr_final'] = 0. 
        df_mc.loc[df_mc[f'{target}_shift']!=0, f'{target}_corr_final'] = (df_mc.loc[df_mc[f'{target}_shift']!=0, f'{target}_shift'] 
            + predict(df_mc.loc[df_mc[f'{target}_shift']!=0, features], model_file, trans_file_corr_diff, target_name))


    if EBEE == 'EE':
        df_mc.loc[:,variables] = transform(df_mc.loc[:,variables], transformer_file, variables)

        df_mc.loc[np.abs(df_mc['probeScEta_orignal'])<=1.653, '{}_corr_final'.format(preshower[0])] = 0.
        df_mc.loc[np.abs(df_mc['probeScEta_orignal'])>1.653, '{}_corr_final'.format(preshower[0])] = (
            df_mc.loc[np.abs(df_mc['probeScEta_orignal'])>1.653, preshower[0]]
            + predict(
                df_mc.loc[np.abs(df_mc['probeScEta_orignal'])>1.653, kinrho+variables+preshower], 
                '{}/mc_{}_{}_final'.format(modeldir, EBEE, preshower[0]), 
                f'mc_{EBEE}', 
                '{}_corr_diff'.format(preshower[0]), 
                )
            )
        df_mc.loc[:,variables] = inverse_transform(df_mc.loc[:,variables], transformer_file, variables)


    df_mc.loc[:,kinrho] = inverse_transform(df_mc.loc[:,kinrho], transformer_file, kinrho)

    print(df_mc.keys())
    print(df_mc)

#    df_mc.to_hdf('{}/df_mc_{}_all_corr_final.h5'.format(outdir,EBEE),'df',mode='w',format='t')
#    df_mc.to_hdf('{}/df_mc_{}_Iso_{}_corr_final.h5'.format(outdir,EBEE,data_type),'df',mode='w',format='t')

    if var_type == 'all': 
        df_mc.to_hdf(f'{outdir}/{inputmc}','df',mode='w',format='t')
    else:
        df_mc.to_hdf(f'{outdir}/{inputmc}'.replace('.h5', '_final.h5'),'df',mode='w',format='t')
 




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-t','--data_type', action='store', type=str)
    optArgs.add_argument('-v','--var_type', action='store', type=str)
    optArgs.add_argument('-s','--split', action='store', type=int)
    options = parser.parse_args()
    main(options)
