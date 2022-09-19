#!/bin/bash
#SBATCH --job-name=avg-l pickles/1119_clcl.obj1367 90 533 1119 1244 1913 1934 2098 2294 1471
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=short

cd /home/unibas/boittier/fdcm_python

python fdcm_python.py  -n 1367 90 533 1119 1244 1913 1934 2098 2294 1471 -l pickles/1119_clcl.obj

