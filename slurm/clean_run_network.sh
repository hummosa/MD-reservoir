#!/bin/bash
#SBATCH -t 00:25:00
#SBATCH -N 1
#SBATCH -c 1 
#SBATCH --mem=12G
#SBATCH --job-name=$1
#SBATCH --output=./slurm/%j
python run_network.py $1 