#!/bin/bash
# init
# function pause(){ read -t 60 -p "Press [Enter] key to run the next model run..." ; }
# Function to pause for a number of seconds seconds
# 1. 2. 5. 10. 15. 
#module load openmind/anaconda/3-2019.10
for i in 1. 2. 5. 10. 15. 20. 25. 30. 35. 40. 50. 60. 75. 90. 100. 200. 400. #MD ampflication
do
   for j in 6.   # PFC_G
    do
      python test_reservoir_PFCMD.py $1 $i $j
      # read -t 60 -p "Press [Enter] key to run the next model run..."
    done
done

