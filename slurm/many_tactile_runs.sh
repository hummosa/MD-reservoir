#!/bin/bash
# rm ./output/*
for s in {0..20} # seeds
do
  for i in 1 #var1
  # for i in 1. 5. 10. 15. 20. 25. 30. 35. 40. 45. 50. 55. 60. #MD ampflication
  # for i in  1. .2 1.3 1.5 #0.2 0.4 0.5 0.7 0.9 G
  do
    for j in 1.0 # Var2
      do
      for k in  0.0 #1.0 # 5. 30. #.1 0.3 0.4 0.6 0.7 1. 1.2 1.5 #Var3
        do
          sbatch ./slurm/run_tactile.sh $1 $s $i $j $k
      done
    done
  done
done