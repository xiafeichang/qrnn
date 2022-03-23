#! /bin/bash

#SBATCH --job-name=BayesOpt 
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu                                           
#SBATCH --nodes=1                        # request to run job on single node                                       
#SBATCH --ntasks=20                      # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)      
#SBATCH --gres=gpu:4                     # request  for two GPU's on machine, this is total  amount of GPUs for job        
#SBATCH --mem=10G                         # memory (per job)
#SBATCH --time=2-00:00                   # time  in format DD-HH:MM
##SBATCH --array=0-5

#python optimize.py -i ${SLURM_ARRAY_TASK_ID}
python optimize.py -i $1

