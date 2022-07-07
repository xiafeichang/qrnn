import argparse
import pandas as pd
from time import time

from mylib.Corrector import Corrector, applyCorrection
from mylib.Shifter import Shifter, applyShift
from mylib.Shifter2D import Shifter2D, apply2DShift
from mylib.transformer import transform, inverse_transform
from mylib.IdMVAComputer import helpComputeIdMva
from mylib.tools import *




def main(options):
    variables = ['probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
    isoVarsPh = ['probePhoIso']
    isoVarsCh = ['probeChIso03','probeChIso03worst']
    preshower = ['probeesEnergyOverSCRawEnergy']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    
    EBEE = options.EBEE
    data_type = options.data_type
    final = True # here final stands for if correct preshower variable and compute photon ID MVA
    
#    df_mc = (pd.read_hdf('weighted_dfs/df_mc_{}_{}.h5'.format(EBEE,data_type))).reset_index(drop=True)
    df_mc = (pd.read_hdf('weighted_dfs/df_mc_{}_Iso_{}.h5'.format(EBEE,data_type))).reset_index(drop=True)

    modeldir = 'test/chained_models'
    outdir = 'test/dfs_corr'
 
    # shift isolation variables
    print(f'shifting mc with classifiers and tail regressor for {isoVarsPh}')
    df_mc['{}_shift'.format(isoVarsPh[0])] = parallelize(applyShift,
        df_mc.loc[:,kinrho], df_mc.loc[:,isoVarsPh[0]],
        load_clf('{}/mc_{}_clf_{}.pkl'.format(modeldir, EBEE, isoVarsPh[0])), 
        load_clf('{}/data_{}_clf_{}.pkl'.format(modeldir, EBEE, isoVarsPh[0])), 
        '{}/mc_{}_{}'.format(modeldir, EBEE, isoVarsPh[0]),
        final_reg = False,
        ) 

    print(f'shifting mc with classifiers and tail regressor for {isoVarsCh}')
    Y_shifted = parallelize(apply2DShift,
        df_mc.loc[:,kinrho], df_mc.loc[:,isoVarsCh],
        load_clf('{}/mc_{}_clf_{}_{}.pkl'.format(modeldir, EBEE, isoVarsCh[0], isoVarsCh[1])), 
        load_clf('{}/data_{}_clf_{}_{}.pkl'.format(modeldir, EBEE, isoVarsCh[0], isoVarsCh[1])), 
        '{}/mc_{}_tReg_{}'.format(modeldir, EBEE, isoVarsCh[0]), 
        '{}/mc_{}_tReg_{}'.format(modeldir, EBEE, isoVarsCh[1]), 
        final_reg = False,
        ) 
    df_mc['{}_shift'.format(isoVarsCh[0])] = Y_shifted[:,0]
    df_mc['{}_shift'.format(isoVarsCh[1])] = Y_shifted[:,1]

     
    # transform features and target to apply qrnn
    df_mc['probeScEta_orignal'] = df_mc['probeScEta']
    transformer_file = 'data_{}'.format(EBEE)
    df_mc.loc[:,kinrho+variables] = transform(df_mc.loc[:,kinrho+variables], transformer_file, kinrho+variables)
    print(df_mc)
    
   
    # correct
    corr_start = time()
    #target = variables[5]
    if EBEE == 'EB' or not final: 
        vars_qrnn = variables.copy() 
    else: 
        vars_qrnn = variables+preshower
    for target in variables: 
        features = kinrho + ['{}_corr'.format(x) for x in variables[:variables.index(target)]]
        
        X = df_mc.loc[:,features]
        Y = df_mc.loc[:,target]
        
        models_mc = '{}/{}_{}_{}'.format(modeldir, 'mc', EBEE, target)
        models_d = '{}/{}_{}_{}'.format(modeldir, 'data', EBEE, target)
        print('Correct {} with features {}'.format(target, features))
        df_mc['{}_corr'.format(target)] = parallelize(applyCorrection, X, Y, models_mc, models_d, diz=False)

    if final:
        print('Correct {} with features {}'.format(preshower[0], features))
        if EBEE == 'EE': 
            features = kinrho + ['{}_corr'.format(x) for x in variables]
#            df_mc.loc[np.abs(df_mc['probeScEta_orignal'])<=1.653, '{}_corr'.format(preshower[0])] = df_mc.loc[np.abs(df_mc['probeScEta_orignal'])<=1.653, preshower[0]]
            df_mc.loc[np.abs(df_mc['probeScEta_orignal'])<=1.653, '{}_corr'.format(preshower[0])] = 0.
            df_mc.loc[np.abs(df_mc['probeScEta_orignal'])>1.653, '{}_corr'.format(preshower[0])] = parallelize(applyCorrection, 
                df_mc.loc[np.abs(df_mc['probeScEta_orignal'])>1.653, features], 
                df_mc.loc[np.abs(df_mc['probeScEta_orignal'])>1.653, preshower[0]], 
                '{}/{}_{}_{}'.format(modeldir, 'mc', EBEE, preshower[0]),
                '{}/{}_{}_{}'.format(modeldir, 'data', EBEE, preshower[0]),
                diz=False, 
                )
        else: 
            df_mc['{}_corr'.format(preshower[0])] = 0.

    
    print('Correct {} with features {}'.format(isoVarsPh[0], kinrho))
    df_mc['{}_corr'.format(isoVarsPh[0])] = parallelize(applyCorrection,
        df_mc.loc[:,kinrho], df_mc.loc[:,'{}_shift'.format(isoVarsPh[0])], 
        '{}/{}_{}_{}'.format(modeldir, 'mc', EBEE, isoVarsPh[0]), 
        '{}/{}_{}_{}'.format(modeldir, 'data', EBEE, isoVarsPh[0]), 
        diz=True, 
        )

    print(f'correcting mc with models for {isoVarsCh}')
    for target in isoVarsCh: 
        features = kinrho + ['{}_corr'.format(x) for x in isoVarsCh[:isoVarsCh.index(target)]]
        
        X = df_mc.loc[:,features]
        Y = df_mc.loc[:,'{}_shift'.format(target)]
        
        models_mc = '{}/{}_{}_{}'.format(modeldir, 'mc', EBEE, target)
        models_d = '{}/{}_{}_{}'.format(modeldir, 'data', EBEE, target)
        print('Correct {} with features {}'.format(target, features))
        df_mc['{}_corr'.format(target)] = parallelize(applyCorrection, X, Y, models_mc, models_d, diz=True)
 
    vars_corr = ['{}_corr'.format(target) for target in variables] 
    df_mc.loc[:,variables+vars_corr] = inverse_transform(df_mc.loc[:,variables+vars_corr], transformer_file, variables+vars_corr)
    df_mc.loc[:,kinrho] = inverse_transform(df_mc.loc[:,kinrho], transformer_file, kinrho)

    if final:
        if EBEE != 'EB': 
            vars_corr = vars_corr + ['{}_corr'.format(var) for var in preshower]
        isoVars = isoVarsPh+isoVarsCh
        isoVars_shift = ['{}_shift'.format(var) for var in isoVars]
        isoVars_corr = ['{}_corr'.format(var) for var in isoVars]
        print(df_mc.loc[:,kinrho+vars_qrnn+vars_corr+isoVars+isoVars_shift+isoVars_corr])
        print('time spent in correction: {} s'.format(time() - corr_start))

        id_start = time()
        weightsEB = 'phoIDmva_weight/HggPhoId_94X_barrel_BDT_v2.weights.xml'
        weightsEE = 'phoIDmva_weight/HggPhoId_94X_endcap_BDT_v2.weights.xml'
        
        phoIDname = 'probePhoIdMVA'
        print('Compute photon ID MVA for uncorrected mc')
        df_mc[phoIDname] = helpComputeIdMva(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars+preshower, df_mc, 'data', False) # +isoVars 
        print('Compute photon ID MVA for corrected mc')
        df_mc['{}_corr'.format(phoIDname)] = helpComputeIdMva(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars+preshower, df_mc, 'qr', False) # +isoVars 
        print('time spent in computing photon ID MVA: {} s'.format(time() - id_start))

    print(df_mc.keys())
#    df_mc.to_hdf('{}/df_mc_{}_{}_corr.h5'.format(outdir,EBEE,data_type),'df',mode='w',format='t')
    df_mc.to_hdf('{}/df_mc_{}_Iso_{}_corr.h5'.format(outdir,EBEE,data_type),'df',mode='w',format='t')
        



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-t','--data_type', action='store', type=str, required=True)
    options = parser.parse_args()
    main(options)
