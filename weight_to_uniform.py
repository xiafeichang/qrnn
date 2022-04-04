import numpy as np
import pandas as pd


def weight_bin(df, weight, xname, xedges, yname=None, yedges=None):
    if yname is None:
        print('bin: ({}, {}), weight: {}'.format(xedges[0], xedges[1], weight))
        querystr = '{} > {} and {} < {}'.format(xname, xedges[0], xname, xedges[1])
    else:
        print('bin: ({}, {}), ({}, {}), weight: {}'.format(xedges[0], xedges[1], yedges[0], yedges[1], weight))
        querystr = '{} > {} and {} < {} and {} > {} and {} < {}'.format(xname, xedges[0], xname, xedges[1], yname, yedges[0], yname, yedges[1])

    df_bin = df.query(querystr)
    df_bin.loc[:,'ml_weight'] = weight
    return df_bin

def assign_weights(df, weights, xname, xedges, yname=None, yedges=None):
    df_weighted = pd.DataFrame()
    if yname is None: 
        print('assign weights for {} bins'.format(xname))
        for i in range(len(xedges)-1): 
            try: 
                df_bin = weight_bin(df,weights[i],xname,(xedges[i],xedges[i+1]))
            except ValueError: 
                print('empty bin, continue')
                continue
            df_weighted = pd.concat([df_weighted, df_bin], axis=0)
    else:
        print('assign weights for ({}, {}) bins'.format(xname, yname))
        for i in range(len(xedges)-1): 
            for j in range(len(yedges)-1):
                try:
                    df_bin = weight_bin(df,weights[i,j],xname,(xedges[i],xedges[i+1]),yname,(yedges[j],yedges[j+1]))
                except ValueError:
                    print('empty bin, continue')
                    continue
                df_weighted = pd.concat([df_weighted, df_bin],axis=0)

    return df_weighted

def compute_weights(hist):
    height = np.max(hist)
    return np.nan_to_num(height/hist, posinf=height, neginf=0.)



def main():

    xname = 'probePt'
    yname = 'rho'
    bins = 50

#    data_key = 'data'
    EBEE = 'EB'
#    df_type = 'train'
    
    for data_key in ('data', 'mc'): 
        for df_type in ('train', 'test'):

            inputfile = 'df_{}_{}_{}.h5'.format(data_key, EBEE, df_type)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>weighting file: ', inputfile)
            df = pd.read_hdf(inputfile)
            print('orignal dataframe: \n', df)

            hist, xedges, yedges = np.histogram2d(df[xname], df[yname], bins=bins)
            weights = compute_weights(hist)

            df_weighted = assign_weights(df, weights, xname, xedges, yname, yedges)

            print('weighted dataframe: \n', df_weighted)
            df_weighted.to_hdf('weighted_dfs/{}'.format(inputfile),'df',mode='w',format='t')


if __name__ == '__main__':
    main()

