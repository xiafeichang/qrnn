#! /bin/bash


#SBATCH -J train_qrnn
##SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH -t 0-08:00
##SBATCH --partition=long

python $1
#python $1 -d $2 -e $3 

#for data_key in "data" "mc";
#do
#    for EBEE in "EB" "EE";
#    do
#        python $1 -d ${data_key} -e ${EBEE}
#    done
#done
