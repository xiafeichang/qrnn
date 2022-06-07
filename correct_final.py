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
    
#    df_mc = (pd.read_hdf('weighted_dfs/df_mc_{}_{}.h5'.format(EBEE,data_type))).reset_index(drop=True)
#    df_mc = (pd.read_hdf('weighted_dfs/df_mc_{}_Iso_{}.h5'.format(EBEE,data_type))).reset_index(drop=True)

#    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_{}_corr.h5'.format(EBEE,data_type))).reset_index(drop=True)
    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_Iso_{}_corr.h5'.format(EBEE,data_type))).reset_index(drop=True)

    modeldir = 'test/chained_models'
    outdir = 'test/dfs_corr'
 
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

#    df_mc.to_hdf('{}/df_mc_{}_{}_corr_final.h5'.format(outdir,EBEE,data_type),'df',mode='w',format='t')
    df_mc.to_hdf('{}/df_mc_{}_Iso_{}_corr_final.h5'.format(outdir,EBEE,data_type),'df',mode='w',format='t')
 




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-t','--data_type', action='store', type=str, required=True)
    options = parser.parse_args()
    main(options)
