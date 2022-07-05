#! /bin/bash

#SBATCH --job-name=qrnn 
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu                                           
#SBATCH --nodes=1                        # request to run job on single node                                       
#SBATCH --ntasks=5                      # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)      
#SBATCH --gres=gpu:1                     # request  for two GPU's on machine, this is total  amount of GPUs for job        
#SBATCH --mem=32G                        # memory (per job)
#SBATCH --time=3-12:00                   # time  in format DD-HH:MM
##SBATCH --nodelist=t3gpu02               # submit to a specific node
#SBATCH --gres-flags=disable-binding    

#nEvt=3500000
#nEvt=1000000

begin=$(date +%s)

ispl=$1

EBEE=(EB EE)
nEvt=(3600000 1000000)
#nEvt=(3200000 970000) # for syst. uncertainty
for i in ${!EBEE[@]}; 
do
    python train_SS.py -e ${EBEE[i]} -n ${nEvt[i]} #-s ${ispl}
    python train_SS_mc.py -e ${EBEE[i]} -n ${nEvt[i]} -r yes #-s ${ispl} 
done

for EBEE in "EB" "EE"; 
do 
    for data_type in "test" "train"; 
    do
        echo correcting mc for ${EBEE} ${data_type}
        python correct_mc.py -e ${EBEE} -t ${data_type} -v 'SS' #-s ${ispl}
    done
done

python train_preshower.py #-s ${ispl}
python train_preshower_mc.py -r yes #-s ${ispl}

nEvt=(3500000 1800000)
#nEvt=(3100000 900000)
for i in ${!EBEE[@]};
do
    python train_Iso.py -e ${EBEE[i]} -n ${nEvt[i]} -v Ph #-s ${ispl}
    python train_Iso_mc.py -e ${EBEE[i]} -n ${nEvt[i]} -v Ph -r yes #-s ${ispl}
    python train_Iso.py -e ${EBEE[i]} -n ${nEvt[i]} -v Ch #-s ${ispl}
    python train_Iso_mc.py -e ${EBEE[i]} -n ${nEvt[i]} -v Ch -r yes #-s ${ispl}
done

for EBEE in "EB" "EE"; 
do 
    for data_type in "test" "train"; 
    do
        echo correcting mc for ${EBEE} ${data_type}
        python correct_mc.py -e ${EBEE} -t ${data_type} -v 'Iso' #-s ${ispl}
    done
done

EBEE=(EB EE)
nEvt=(3600000 1000000)
#nEvt=(3200000 970000) # for syst. uncertainty
for i in ${!EBEE[@]}; 
do
    python train_final_SS.py -e ${EBEE[i]} -n ${nEvt[i]} #-s ${ispl} 
done

nEvt=(3500000 1800000)
#nEvt=(3100000 900000)
for i in ${!EBEE[@]};
do
    python train_final_Iso.py -e ${EBEE[i]} -n ${nEvt[i]} -v Ph #-s ${ispl} 
    python train_final_Iso.py -e ${EBEE[i]} -n ${nEvt[i]} -v Ch #-s ${ispl} 
done

python train_final_preshower.py #-s ${ispl}

#for EBEE in "EB" "EE"; 
#do 
#    for data_type in "test" "train"; 
#    do
#        echo correcting mc for ${EBEE} ${data_type}
#        python correct_final.py -e ${EBEE} -t ${data_type} #-s ${ispl} 
#        python correct_final_Iso.py -e ${EBEE} -t ${data_type} #-s ${ispl} 
#    done
#done

for EBEE in "EB" "EE";
do
    python correct_mc.py -e ${EBEE} -v "all" #-s ${ispl}
    python correct_final.py -e ${EBEE} -v "all" #-s ${ispl} 
    python correct_final_Iso.py -e ${EBEE} -v "all" #-s ${ispl} 
done

EBEE=(EB EE)
nEvt=(7000000 2000000)
for i in ${!EBEE[@]};
do
    python check_results.py -e ${EBEE[i]} -n ${nEvt[i]}
done

tottime=$(echo "$(date +%s) - $begin" | bc)
echo ">>>>>>>>>>>>>>>>>>>> time spent: $tottime s" 

