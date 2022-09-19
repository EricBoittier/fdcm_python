#!/bin/bash
#SBATCH --job-name=avg1 335 1481 1864 2081 2418
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=short

cd /home/unibas/boittier/fdcm_python

python fdcm_python.py -f -n 1 335 1481 1864 2081 2418 


sbatch jobs/p1906_1119.sh 

sbatch jobs/p1906_2037.sh 
