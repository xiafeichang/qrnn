import argparse
import pandas as pd
from qrnn import QRNN

def main(options):

    kinrho = ['probePt','probeScEta','probePhi','rho'] 
      
    inputfile = 'df_data_EB_train.h5'
    n_evt = 2000000
    
    #load dataframe
    df_total = pd.read_hdf(inputfile)
    df_smp = df_total.sample(n_evt, random_state=100).reset_index(drop=True)
    
    df_train = df_smp[:1000000] 
    
    #set features and target
    features = kinrho 
    target = options.variable
    
    num_hidden_layers = 5
    num_units = [1000, 500, 100]
    act = ['tanh', 'softplus', 'elu']
    
    scale_para_file = 'scale_para/{}_{}.h5'.format(inputfile[3:-12], target)
    
    q = options.quantile
    model_file = 'models/{}_{}'.format(target,str(q).replace('.','p'))
    
    qrnn = QRNN(df_train, features, target, scale_file=scale_para_file)
    qrnn.trainQuantile(q,num_hidden_layers,num_units,act,batch_size=8192, save_file=model_file)
    

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-q','--quantile', action='store', type=float, required=True)
    requiredArgs.add_argument('-v','--variable', action='store', type=str, required=True)

