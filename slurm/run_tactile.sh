#!/bin/bash
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH -c 1 
#SBATCH --mem=12G
#SBATCH --job-name=$1
#SBATCH --output=./output/%j
python run_network.py $1 $2 --var1=$3 --var2=$4 --var3=$5
