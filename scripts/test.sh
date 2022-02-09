#! /bin/bash

var=$1
q=$2

#SBATCH -J ${var}-${q}
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -t 00:30:00

python test.py -v ${var} -q ${q}

