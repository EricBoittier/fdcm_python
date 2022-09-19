#!/bin/bash
#SBATCH --job-name=/data/unibas/boittier/methanol_gs_200.t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=short

cd /home/unibas/boittier/fdcm_python

python fdcm_python.py  -n 197 105 296 310 557 728 738 1234 1564 1649 -l pickles/1234_clcl.obj


sbatch jobs/p7_446.sh 

sbatch jobs/p7_18.sh 
