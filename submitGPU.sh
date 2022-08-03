#! /bin/bash

#SBATCH --job-name=qrnn 
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu                                           
#SBATCH --nodes=1                        # request to run job on single node                                       
#SBATCH --ntasks=10                      # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)      
#SBATCH --gres=gpu:2                     # request  for two GPU's on machine, this is total  amount of GPUs for job        
#SBATCH --mem=40G                        # memory (per job)
#SBATCH --time=3-12:00                   # time  in format DD-HH:MM
##SBATCH --nodelist=t3gpu02               # submit to a specific node
#SBATCH --gres-flags=disable-binding    
##SBATCH --array=0-5

EBEE=$1
nEvt=$2

ispl=$3

#EBEE=EB
#nEvt=3600000
#nEvt=3500000
#nEvt=1000000

#python train_preshower.py
#python train_preshower_mc.py -r yes

#python train_SS.py -d data -e ${EBEE} -n ${nEvt} -i ${SLURM_ARRAY_TASK_ID}
#python train_SS_mc.py -e ${EBEE} -n ${nEvt} -r yes 

#python train_Iso.py -e ${EBEE} -n ${nEvt} -v Ph #-s ${ispl}
#python train_Iso_mc.py -e ${EBEE} -n ${nEvt} -v Ph -r yes #-s ${ispl}
#python train_Iso.py -e ${EBEE} -n ${nEvt} -v Ch #-s ${ispl}
#python train_Iso_mc.py -e ${EBEE} -n ${nEvt} -v Ch -r yes #-s ${ispl}
#python check_results.py -e ${EBEE} -n ${nEvt}

#EBEE=(EB EE)
#nEvt=(3600000 1000000)
#for i in ${!EBEE[@]}; 
#do
#    python train_final_SS.py -e ${EBEE[i]} -n ${nEvt[i]} 
#done

#nEvt=(3500000 1800000)
#nEvt=(3100000 900000)
#for i in ${!EBEE[@]}; 
#do
#    python train_final_Iso.py -e ${EBEE[i]} -n ${nEvt[i]} -v Ph  #-s ${ispl}
#    python train_final_Iso.py -e ${EBEE[i]} -n ${nEvt[i]} -v Ch  #-s ${ispl}
#done

#python train_final_preshower.py

#for EBEE in "EE"; # "EE" "EB" ; 
#do 
#    for data_type in "test" "train"; 
#    do
#        echo correcting mc for ${EBEE} ${data_type}
#        #python correct_mc.py -e ${EBEE} -t ${data_type} 
#        python correct_final.py -e ${EBEE} -t ${data_type} 
#        #python correct_final_Iso.py -e ${EBEE} -t ${data_type} 
#        #python correct_final_uncer.py -e ${EBEE} -t ${data_type} 
#    done
#done

python correct_mc.py -e ${EBEE} -t train -v $2 -f yes #-s ${ispl}
#python correct_final_Iso.py -e ${EBEE} -v all #-s ${ispl} 

#python train_final_SS_uncer.py -e ${EBEE} -n ${nEvt} 
#python correct_final_uncer.py -e ${EBEE} -t $2

#python test_pred.py -e ${EBEE} -n ${nEvt} -v Ch


