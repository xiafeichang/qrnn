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
    
#    df_mc = (pd.read_hdf('weighted_dfs/df_mc_{}_Iso_{}.h5'.format(EBEE,data_type))).reset_index(drop=True)
    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_Iso_{}_corr.h5'.format(EBEE,data_type))).reset_index(drop=True)
#    df_mc = (pd.read_hdf('dfs_corr/df_mc_{}_Iso_{}_corr_final.h5'.format(EBEE,data_type))).reset_index(drop=True)

    modeldir = 'chained_models'
    outdir = 'dfs_corr'
 
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
    
    df_mc.loc[:,kinrho] = inverse_transform(df_mc.loc[:,kinrho], transformer_file, kinrho)

    print(df_mc.keys())
    print(df_mc)

    df_mc.to_hdf('{}/df_mc_{}_Iso_{}_corr_final.h5'.format(outdir,EBEE,data_type),'df',mode='w',format='t')
 




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-t','--data_type', action='store', type=str, required=True)
    options = parser.parse_args()
    main(options)
