#! /bin/bash


#SBATCH -J try_qrnn
##SBATCH -n 1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH -t 0-08:00
##SBATCH --partition=long

#python try_qrnn.py
#python compare_data_mc.py
#python check_results.py
#python transform.py
#python $1 -i $2
python $1
