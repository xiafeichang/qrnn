#! /bin/bash

#SBATCH --job-name=qrnn 
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu                                           
#SBATCH --nodes=1                        # request to run job on single node                                       
#SBATCH --ntasks=10                      # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)      
#SBATCH --gres=gpu:2                     # request  for two GPU's on machine, this is total  amount of GPUs for job        
#SBATCH --mem=40G                        # memory (per job)
#SBATCH --time=2-00:00                   # time  in format DD-HH:MM
##SBATCH --nodelist=t3gpu01               # submit to a specific node
#SBATCH --gres-flags=disable-binding    
##SBATCH --array=0-5

#python $1 -d $2 -e $3 -i ${SLURM_ARRAY_TASK_ID}
#python $1 -e $2 -t $3 
#python $1 -e $2 
#python $1 -r yes

EBEE=EB
nEvt=3500000

#python train_Iso.py -e ${EBEE} -n ${nEvt} -v Ph
#python train_Iso_mc.py -e ${EBEE} -n ${nEvt} -v Ph -r yes
#python train_Iso.py -e ${EBEE} -n ${nEvt} -v Ch
#python train_Iso_mc.py -e ${EBEE} -n ${nEvt} -v Ch -r yes
python check_results.py -e ${EBEE} -n ${nEvt}

