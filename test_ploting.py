from check_results import * 


nEvnt = 3500000

df_data = (pd.read_hdf('/work/xchang/tmp/from_massi/df_data_EB_Iso_test.h5')).sample(nEvnt, random_state=100).reset_index(drop=True)
df_mc = (pd.read_hdf('/work/xchang/tmp/from_massi/df_mc_EB_Iso_test.h5')).sample(nEvnt, random_state=100).reset_index(drop=True)

kinrho = ['probePt','probeScEta','probePhi','rho'] 
transformer_file = 'data_EB'
df_mc.loc[:,kinrho] = transform(df_mc.loc[:,kinrho], transformer_file, kinrho)

df_mc['weight_clf'] = clf_reweight(df_mc, df_data, 'transformer/4d_reweighting_EB', n_jobs=10)


target = 'probeChIso03worst'
draw_hist(df_data, df_mc, nEvnt, target, f'plots/other_test/data_mc_dist_EB_{target}', bins=75, histrange=(0., 10.), density=True, mc_weights=df_mc['weight_clf'])


