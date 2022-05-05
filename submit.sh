#! /bin/bash


#SBATCH -J train_qrnn
##SBATCH -n 1
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH -t 0-08:00
##SBATCH --partition=long

#python $1
#python $1 -d $2 -e $3 

python make_dataframes.py -d data -e EB
python make_dataframes.py -d mc -e EB
python make_dataframes.py -d data -e EE
python make_dataframes.py -d mc -e EE

python weight_to_uniform.py -d data -e EB
python weight_to_uniform.py -d mc -e EB
python weight_to_uniform.py -d data -e EE
python weight_to_uniform.py -d mc -e EE

#python train_Iso.py -e EB -v Ph
#python train_Iso_mc.py -e EB -v Ph -r yes
#python train_Iso.py -e EB -v Ch
#python train_Iso_mc.py -e EB -v Ch -r yes
#python check_results.py -e EB

