#!/bin/bash
#SBATCH --job-name=avg-l pickles/1906_clcl.obj1119 1906 2037 2294 1367
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=short

cd /home/unibas/boittier/fdcm_python

python fdcm_python.py  -n 1119 1906 2037 2294 1367 -l pickles/1906_clcl.obj


sbatch jobs/p1367_1471.sh 
