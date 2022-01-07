#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH --output=/om2/group/halassa/MD-reservoir_res-sab/hebbian_810/stdout/%j
#SBATCH --job-name=$1
#SBATCH --array=1-810%10
python run_network.py $1 $2 --var1 $SLURM_ARRAY_TASK_ID 
