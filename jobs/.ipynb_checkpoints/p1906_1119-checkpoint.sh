#!/bin/bash
#SBATCH --job-name=avg-l pickles/1_clcl.obj1906 1 2037 2081
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=short

cd /home/unibas/boittier/fdcm_python

python fdcm_python.py  -n 1906 1 2037 2081 -l pickles/1_clcl.obj


sbatch jobs/p1119_1367.sh 

sbatch jobs/p1119_2294.sh 
