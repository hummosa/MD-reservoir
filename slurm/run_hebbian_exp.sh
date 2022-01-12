#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH --output=./output/%j
#SBATCH --job-name=$1
#SBATCH --array=1-200%100
python run_network.py $1 --slurm_task_id $SLURM_ARRAY_TASK_ID --var3 4
