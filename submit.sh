#! /bin/bash


#SBATCH -J try_qrnn
#SBATCH -n 1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH -t 0-08:00
##SBATCH --partition=long

#python try_qrnn.py
python compare_data_mc.py

