#! /bin/bash

#SBATCH --job-name=qrnn 
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu                                           
#SBATCH --nodes=1                        # request to run job on single node                                       
#SBATCH --ntasks=5                       # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)      
#SBATCH --gres=gpu:1                     # request  for two GPU's on machine, this is total  amount of GPUs for job        
#SBATCH --mem=10G                        # memory (per job)
#SBATCH --time=2-00:00                   # time  in format DD-HH:MM
##SBATCH --nodelist=t3gpu01               # submit to a specific node
#SBATCH --gres-flags=disable-binding    
##SBATCH --array=0-5

#python try_qrnn.py -i ${SLURM_ARRAY_TASK_ID}
#python try_qrnn_mc.py
#python compare_data_mc.py
#python check_results.py
#python check_gpu.py
#python $1 -d $2 -e $3 -i ${SLURM_ARRAY_TASK_ID}
python $1

#for q in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99;
#for q in 0.03 0.97;
#do 
#    echo for quantile ${q}
#    python try_qrnn2.py -i 0 -q ${q}
#done

