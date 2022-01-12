#!/bin/bash
# rm ./output/*
for s in {0..10} # seeds
do
  for i in 1 #var1
  # for i in 1. 5. 10. 15. 20. 25. 30. 35. 40. 45. 50. 55. 60. #MD ampflication
  # for i in  1. .2 1.3 1.5 #0.2 0.4 0.5 0.7 0.9 G
  do
    # for j in 0 1 2 3 4 5 10 20 30 40 50 100 200 # Var2  timesteps
    # for j in 0 10 20 30 50 100 200 300 400 500 # Var2  no of dlPFC neurons
    for j in 0  # Var2  no of dlPFC neurons
      do
      for k in  3 #1.0 # 5. 30. #.1 0.3 0.4 0.6 0.7 1. 1.2 1.5 #Var3
        do
          sbatch ./slurm/run_tactile.sh $1 $s $i $j $k
      done
    done
  done
done