#! /bin/bash

#SBATCH --job-name=qrnn 
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu                                           
#SBATCH --nodes=1                        # request to run job on single node                                       
#SBATCH --ntasks=10                      # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)      
#SBATCH --gres=gpu:2                     # request  for two GPU's on machine, this is total  amount of GPUs for job        
#SBATCH --mem=5G                         # memory (per job)
#SBATCH --time=2-00:00                   # time  in format DD-HH:MM

#python try_qrnn.py
python try_qrnn_mc.py

