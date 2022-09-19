#!/bin/bash
#SBATCH --job-name=/data/unibas/boittier/methanol_gs_200.t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=short

cd /home/unibas/boittier/fdcm_python

python fdcm_python.py  -n -n 2434 1019 1422 1925 -l pickles/1019_clcl.obj

