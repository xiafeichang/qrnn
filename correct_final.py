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
            iptdir = f'dfs_sys/split{spl}'
            inputmc = f'df_mc_{EBEE}_all_corr.h5'
        else:
            iptdir = f'dfs_sys/split{spl}'
            inputmc = f'df_mc_{EBEE}_{data_type}_split{spl}_corr.h5'
    else: 
        if var_type == 'all': 
            iptdir = f'dfs_corr'
            inputmc = f'df_mc_{EBEE}_all_corr.h5'
        else:
            iptdir = 'dfs_corr'
            inputmc = f'df_mc_{EBEE}_{data_type}_corr.h5'
        warnings.warn(f"Wrong argument '-s' ('--split'), argument must have value 1 or 2. Now using defalt dataframe {iptdir}/{inputmc}")
    if data_type == 'test': 
        inputmc = inputmc.replace(f'_split{spl}','')
       
    print(f'loading dataframe from {iptdir}/{inputmc}')
    df_mc = (pd.read_hdf(f'{iptdir}/{inputmc}')).reset_index(drop=True)

#    df_mc = (pd.read_hdf('weighted_dfs/df_mc_{}_{}.h5'.format(EBEE,data_type))).reset_index(drop=True)
#    df_mc = (pd.read_hdf('weighted_dfs/df_mc_{}_Iso_{}.h5'.format(EBEE,data_type))).reset_index(drop=True)

#    df_mc = (pd.read_hdf(f'dfs_corr/df_mc_{EBEE}_all_corr.h5')).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_{}_corr.h5'.format(EBEE,data_type))).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_Iso_{}_corr.h5'.format(EBEE,data_type))).reset_index(drop=True)

    if spl in [1,2]:
        modeldir = f'models/split{spl}'
        outdir = f'dfs_sys/split{spl}'
    else:
        modeldir = 'chained_models'
        outdir   = 'dfs_corr'
    print(f'using models from {modeldir}, corrected dataframes will be saved in {outdir}')
 
    transformer_file = 'data_{}'.format(EBEE)
    df_mc.loc[:,kinrho+variables] = transform(df_mc.loc[:,kinrho+variables], transformer_file, kinrho+variables)

    model_file = '{}/mc_{}_SS_final'.format(modeldir, EBEE)
    trans_file_corr_diff = f'mc_{EBEE}'

    features = kinrho + variables
    target_name = [f'{var}_corr_diff' for var in variables]
    df_diff = predict(df_mc.loc[:, features], model_file, trans_file_corr_diff, target_name) 
    print(df_diff)

    df_mc.loc[:,kinrho+variables] = inverse_transform(df_mc.loc[:,kinrho+variables], transformer_file, kinrho+variables)

    for var in variables: 
        df_mc[f'{var}_corr_final'] = df_mc[var] + df_diff[f'{var}_corr_diff']
    print(df_mc.keys())
    print(df_mc)

#    df_mc.to_hdf('{}/df_mc_{}_all_corr_final.h5'.format(outdir,EBEE),'df',mode='w',format='t')
#    df_mc.to_hdf('{}/df_mc_{}_{}_corr_final.h5'.format(outdir,EBEE,data_type),'df',mode='w',format='t')
#    df_mc.to_hdf('{}/df_mc_{}_Iso_{}_corr_final.h5'.format(outdir,EBEE,data_type),'df',mode='w',format='t')

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
