import warnings 
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
#    isoVarsCh = ['probeChIso03','probeChIso03worst']
    isoVarsCh = ['probeChIso03worst','probeChIso03']
    preshower = ['probeesEnergyOverSCRawEnergy']
    kinrho = ['probePt','probeScEta','probePhi','rho'] 
    
    EBEE = options.EBEE
    data_type = options.data_type
    print('data_type: ', data_type)
    var_type = options.var_type
    spl = options.split

    if options.final is not None:
        if options.final.lower() == 'yes':
            final = True # here final stands for if correct preshower variable and compute photon ID MVA
        else: 
            final = False
    else: 
        final = False
    
    if var_type == 'all': 
#        iptdir = 'tmp_dfs/all'
#        inputmc = f'df_mc_{EBEE}_all.h5'
        iptdir = 'dfs_sys/backup'
        inputmc = f'df_mc_{EBEE}_all_corr_final.h5'
    elif spl in [1, 2]: 
        iptdir = 'tmp_dfs/weightedsys'
        inputmc = f'df_mc_{EBEE}_Iso_{data_type}_split{spl}.h5'
    else: 
        iptdir = 'tmp_dfs/weighted0.9'
        inputmc = f'df_mc_{EBEE}_Iso_{data_type}.h5'
        warnings.warn(f'Wrong argument "-s" ("--split"), argument must have value 1 or 2. Now using defalt dataframe {iptdir}/{inputmc}')
    if data_type == 'test': 
        inputmc = inputmc.replace(f'_split{spl}','')
    if var_type == 'SS':
        inputmc = inputmc.replace('_Iso','')
        
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>correcting file {iptdir}/{inputmc}')
    df_mc = (pd.read_hdf(f'{iptdir}/{inputmc}')).reset_index(drop=True)

#    df_mc = (pd.read_hdf(f'tmp_dfs/weighted0.9/df_mc_{EBEE}_Iso_{data_type}.h5')).reset_index(drop=True)
#    df_mc = (pd.read_hdf(f'tmp_dfs/all/df_mc_{EBEE}_all.h5')).reset_index(drop=True)
#    df_mc = (pd.read_hdf(f'dfs_corr/df_mc_{EBEE}_all_corr.h5')).reset_index(drop=True)
#    df_mc = (pd.read_hdf('weighted_dfs/df_mc_{}_{}.h5'.format(EBEE,data_type))).reset_index(drop=True)
#    df_mc = (pd.read_hdf('weighted_dfs/df_mc_{}_Iso_{}.h5'.format(EBEE,data_type))).reset_index(drop=True)

    if spl in [1,2]:
        print('correcting for systmetic uncertainty')
        modeldir = f'models/split{spl}'
        outdir = f'dfs_sys/split{spl}'
    else:
        modeldir = 'chained_models'
#        outdir   = 'dfs_corr'
        outdir   = 'dfs_sys'

    print(f'using models from {modeldir}, corrected dataframes will be saved in {outdir}')
 
   
    if var_type == 'Iso' or var_type == 'all': 
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
        # VERY IMPORTANT! Note the order of targets here
        Y_shifted = parallelize(apply2DShift,
            df_mc.loc[:,kinrho], df_mc.loc[:,['probeChIso03','probeChIso03worst']],
            load_clf('{}/mc_{}_clf_{}_{}.pkl'.format(modeldir, EBEE, isoVarsCh[0], isoVarsCh[1])), 
            load_clf('{}/data_{}_clf_{}_{}.pkl'.format(modeldir, EBEE, isoVarsCh[0], isoVarsCh[1])), 
            '{}/mc_{}_tReg_probeChIso03'.format(modeldir, EBEE), 
            '{}/mc_{}_tReg_probeChIso03worst'.format(modeldir, EBEE), 
            final_reg = False,
            ) 
        df_mc['probeChIso03_shift'] = Y_shifted[:,0]
        df_mc['probeChIso03worst_shift'] = Y_shifted[:,1]

         
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

#    if var_type == 'SS' or var_type == 'all':
#        for target in variables: 
#            features = kinrho + ['{}_corr'.format(x) for x in variables[:variables.index(target)]]
#            
#            X = df_mc.loc[:,features]
#            Y = df_mc.loc[:,target]
#            
#            models_mc = '{}/{}_{}_{}'.format(modeldir, 'mc', EBEE, target)
#            models_d = '{}/{}_{}_{}'.format(modeldir, 'data', EBEE, target)
#            print('Correct {} with features {}'.format(target, features))
#            df_mc['{}_corr'.format(target)] = parallelize(applyCorrection, X, Y, models_mc, models_d, diz=False)
#
##        if final:
#        print('Correct {} with features {}'.format(preshower[0], features))
#        if EBEE == 'EE': 
#            features = kinrho + ['{}_corr'.format(x) for x in variables]
##            df_mc.loc[np.abs(df_mc['probeScEta_orignal'])<=1.653, '{}_corr'.format(preshower[0])] = df_mc.loc[np.abs(df_mc['probeScEta_orignal'])<=1.653, preshower[0]]
#            df_mc.loc[np.abs(df_mc['probeScEta_orignal'])<=1.653, '{}_corr'.format(preshower[0])] = 0.
#            df_mc.loc[np.abs(df_mc['probeScEta_orignal'])>1.653, '{}_corr'.format(preshower[0])] = parallelize(applyCorrection, 
#                df_mc.loc[np.abs(df_mc['probeScEta_orignal'])>1.653, features], 
#                df_mc.loc[np.abs(df_mc['probeScEta_orignal'])>1.653, preshower[0]], 
#                '{}/{}_{}_{}'.format(modeldir, 'mc', EBEE, preshower[0]),
#                '{}/{}_{}_{}'.format(modeldir, 'data', EBEE, preshower[0]),
#                diz=False, 
#                )
#        else: 
#            df_mc['{}_corr'.format(preshower[0])] = 0.

    
    if var_type == 'Iso' or var_type == 'all':
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
    if var_type == 'SS' or var_type == 'all':
        df_mc.loc[:,vars_corr] = inverse_transform(df_mc.loc[:,vars_corr], transformer_file, vars_corr)
    df_mc.loc[:,variables] = inverse_transform(df_mc.loc[:,variables], transformer_file, variables)
    df_mc.loc[:,kinrho] = inverse_transform(df_mc.loc[:,kinrho], transformer_file, kinrho)

    if final:
        if EBEE != 'EB': 
            vars_corr = vars_corr + ['{}_corr'.format(var) for var in preshower]
        isoVars = isoVarsPh+isoVarsCh
        isoVars_shift = ['{}_shift'.format(var) for var in isoVars]
        isoVars_corr = ['{}_corr'.format(var) for var in isoVars]
        if var_type == 'SS':
            print(df_mc.loc[:,kinrho+vars_qrnn+vars_corr])
        elif var_type == 'Iso':
            print(df_mc.loc[:,kinrho+isoVars+isoVars_shift+isoVars_corr])
        elif vars_corr == 'all':
            print(df_mc.loc[:,kinrho+vars_qrnn+vars_corr+isoVars+isoVars_shift+isoVars_corr])
        print('time spent in correction: {} s'.format(time() - corr_start))

        id_start = time()
        weightsEB = 'phoIDmva_weight/HggPhoId_94X_barrel_BDT_v2.weights.xml'
        weightsEE = 'phoIDmva_weight/HggPhoId_94X_endcap_BDT_v2.weights.xml'
        
        phoIDname = 'probePhoIdMVA'
        print('Compute photon ID MVA for uncorrected mc')
        stride = int(df_mc.index.size/10) + 1
        df_mc[phoIDname] = np.concatenate(Parallel(n_jobs=10, verbose=20)(delayed(helpComputeIdMva)(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars, df_mc[ch:ch+stride], 'data', False) for ch in range(0, df_mc.index.size, stride))) # variables+isoVars
#        df_mc[phoIDname] = helpComputeIdMva(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars, df_mc, 'data', False) # +isoVars 
        print('Compute photon ID MVA for corrected mc')
        df_mc['{}_corr'.format(phoIDname)] = np.concatenate(Parallel(n_jobs=10, verbose=20)(delayed(helpComputeIdMva)(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars, df_mc[ch:ch+stride], 'qr', False) for ch in range(0, df_mc.index.size, stride))) # variables+isoVars
#        df_mc['{}_corr'.format(phoIDname)] = helpComputeIdMva(weightsEB, weightsEE, EBEE, vars_qrnn+isoVars, df_mc, 'qr', False) # +isoVars 
        print('time spent in computing photon ID MVA: {} s'.format(time() - id_start))

    print(df_mc.keys())
#    df_mc.to_hdf('{}/df_mc_{}_all_corr.h5'.format(outdir,EBEE),'df',mode='w',format='t')
#    df_mc.to_hdf('{}/df_mc_{}_{}_corr.h5'.format(outdir,EBEE,data_type),'df',mode='w',format='t')
#    df_mc.to_hdf('{}/df_mc_{}_Iso_{}_corr.h5'.format(outdir,EBEE,data_type),'df',mode='w',format='t')

    if inputmc.endswith('_corr.h5') or inputmc.endswith('_corr_final.h5'):
        df_mc.to_hdf(f'{outdir}/{inputmc}','df',mode='w',format='t')
    else: 
        df_mc.to_hdf(f'{outdir}/{inputmc}'.replace('.h5', '_corr.h5'),'df',mode='w',format='t')




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-t','--data_type', action='store', type=str)
    optArgs.add_argument('-v','--var_type', action='store', type=str)
    optArgs.add_argument('-s','--split', action='store', type=int)
    optArgs.add_argument('-f','--final', action='store', type=str)
    options = parser.parse_args()
    main(options)
