import os
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import pickle
import os.path

job_template = """#!/bin/bash
#SBATCH --job-name={n}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=vshort
#SBATCH --output={o}/logs/{n}.log

cd /home/unibas/boittier/fdcm_python

python fdcm_python.py {f} -n {n} {l} -o {o} -l2 {l2} > {o}/output/{n}.out

"""


def safe_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


class Job:
    def __init__(self, graph_pkl, outdir, l2=100.0):
        self.G = pd.read_pickle(graph_pkl)
        self.outdir = outdir
        self.l2 = l2
        self.pickles_path = Path(self.outdir) / "pickles"
        self.output_path = Path(self.outdir) / "output"
        self.logs_path = Path(self.outdir) / "logs"
        self.jobs_path = Path(self.outdir) / "jobs"
        self.make_out_dir()

    def make_jobs(self):
        pass

    def make_out_dir(self):
        safe_mkdir(self.outdir)
        safe_mkdir(self.pickles_path)
        safe_mkdir(self.output_path)
        safe_mkdir(self.logs_path)
        safe_mkdir(self.jobs_path)

    def scan_all(self):
        for node in self.G.nodes:

            with open(self.jobs_path / f"{node}.sh", "w") as f:
                outstr = job_template.format(
                    f="",
                    l="",
                    l2=self.l2,
                    n=node,
                    o=self.outdir
                )
                f.write(outstr)

    def scan_avg(self, path_to_prev, method=np.mean):
        prev = Path(path_to_prev)
        safe_mkdir(Path(self.outdir) / "prev")
        for node in self.G.nodes:
            #  get local charge result from neighbours
            prev_pickles = []
            for edge in self.G.edges(node):
                fname = prev / "pickles" / f"{edge[1]}_clcl.obj"
                if os.path.isfile(fname):
                    prev_pickles.append(fname)

            neighbour_charges = [pd.read_pickle(x) for x in prev_pickles]
            mean_charge_pos = method(neighbour_charges, axis=0)
            outname = Path(self.outdir) / "prev" / f"{node}_clcl.obj"
            filehandler = open(outname, "wb")
            pickle.dump(mean_charge_pos, filehandler)

            with open(self.jobs_path / f"{node}.sh", "w") as f:
                outstr = job_template.format(
                    f="",
                    l=f"-l {outname}",
                    l2=self.l2,
                    n=node,
                    o=self.outdir
                )
                f.write(outstr)




