#!/bin/bash                         # Bash shell
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=12G
#SBATCH --output=./output_test/%j
#SBATCH --job-name=$1
#SBATCH --seed=$2
#SBATCH --array=1-10%10 # Submit 10 jobs, 10 run at the same time
python run_network.py $1 $2 $SLURM_ARRAY_TASK_ID
