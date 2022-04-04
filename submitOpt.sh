#! /bin/bash

#SBATCH --job-name=BayesOpt 
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu                                           
#SBATCH --nodes=1                        # request to run job on single node                                       
#SBATCH --ntasks=10                      # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)      
#SBATCH --gres=gpu:2                     # request  for two GPU's on machine, this is total  amount of GPUs for job        
#SBATCH --mem=10G                        # memory (per job)
#SBATCH --time=3-00:00                   # time  in format DD-HH:MM
#SBATCH --nodelist=t3gpu01               # submit to a specific node
#SBATCH --gres-flags=disable-binding    
##SBATCH --array=0-5

data_key=$1
ith_var=$2

python optimize.py -d ${data_key} -i ${ith_var} -e EB
#python try_qrnn_bayesOpt.py -d ${data_key} -i ${ith_var} -e EB

