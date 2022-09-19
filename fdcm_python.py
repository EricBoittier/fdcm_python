import sys
sys.path.insert(0,"/home/unibas/boittier/fdcm_python/PyDCM-1")
# Basics
import os
import numpy as np
import pickle
import pandas as pd
import argparse
import configparser

# Load FDCM modules
from pydcm import Scan, DCM, mdcm

# Optimization
from scipy.optimize import minimize

mdcm_cxyz = "/home/unibas/boittier/fdcm_project/mdcms/methanol/10charges.xyz"
mdcm_clcl = "/home/unibas/boittier/MDCM/examples/multi-conformer/5-charmm-files/10charges.dcm"

def get_clcl(local_pos, charges):
    NCHARGES = len(charges)
    _clcl_ = np.zeros(NCHARGES)
    for i in range(NCHARGES):
        if (i+1) % 4 == 0:
            _clcl_[i] = charges[i]
        else:
            _clcl_[i] = local_pos[i - ((i)//4)]
    return _clcl_

def set_bounds(local_pos, change=0.1):
    bounds = []
    for i, x in enumerate(local_pos):
        bounds.append((x - abs(x)*change, x + abs(x)*change))
    return tuple(bounds)


def mdcm_set_up(nodes_to_average, first=True, local_pos=None):
    scan_fesp = []
    scan_fdns = []
    
    if first==True:
        # Prepare some cube file list
        scan_fesp = [
            "/home/unibas/boittier/MDCM/examples/multi-conformer/ref/scan0.p.cube"
        ]
        scan_fdns = [
            "/home/unibas/boittier/MDCM/examples/multi-conformer/ref/scan0.d.cube"
        ]


    for node in nodes_to_average:
        scan_fesp.append(f"/data/unibas/boittier/graphscan/methanol/t3/p{node}.p.cube")
        scan_fdns.append(f"/data/unibas/boittier/graphscan/methanol/t3/p{node}.d.cube")

    Nfiles = len(scan_fesp)
    Nchars = int(np.max([
        len(filename) for filelist in [scan_fesp, scan_fdns] 
        for filename in filelist]))

    esplist = np.empty([Nfiles, Nchars], dtype='c')
    dnslist = np.empty([Nfiles, Nchars], dtype='c')

    for ifle in range(Nfiles):
        esplist[ifle] = "{0:{1}s}".format(scan_fesp[ifle], Nchars)
        dnslist[ifle] = "{0:{1}s}".format(scan_fdns[ifle], Nchars)

    # Load cube files, read MDCM global and local files
    mdcm.load_cube_files(Nfiles, Nchars, esplist.T, dnslist.T)
    mdcm.load_clcl_file(mdcm_clcl)
    mdcm.load_cxyz_file(mdcm_cxyz)

    # Write MDCM global from local and Fitted ESP cube files
    mdcm.write_cxyz_files()
    mdcm.write_mdcm_cube_files()

    # Get and set local MDCM array (to check if manipulation is possible)
    clcl = mdcm.mdcm_clcl
    mdcm.set_clcl(clcl)
    clcl = mdcm.mdcm_clcl
    
    if local_pos is not None:
        mdcm.set_clcl(local_pos)
        clcl = mdcm.mdcm_clcl
        
    # Get and set global MDCM array (to check if manipulation is possible)
    cxyz = mdcm.mdcm_cxyz
    mdcm.set_cxyz(cxyz)
    cxyz = mdcm.mdcm_cxyz
    return mdcm


def optimize_mdcm(mdcm):
    # Get RMSE, averaged or weighted over ESP files, or per ESP file each
    rmse = mdcm.get_rmse()
    print(rmse)
    srmse = mdcm.get_rmse_each(Nfiles)
    print(srmse)

    #  save an array containing original charges
    charges = clcl.copy()
    local_pos = clcl[np.mod(np.arange(clcl.size)+1, 4) != 0]

    local_ref = local_pos.copy()
    
    
    """Minimization routine"""
    def mdcm_rmse(local_pos, local_ref=local_ref, l2=100.):
        _clcl_ = get_clcl(local_pos, charges)
        mdcm.set_clcl(_clcl_)
        rmse = mdcm.get_rmse()
        if local_ref is not None:
            l2diff = l2*np.sum((local_pos - local_ref)**2)/local_pos.shape[0]
            # print(rmse, l2diff)
            rmse += l2diff 
        return rmse

    # Apply simple minimization without any feasibility check (!)
    # Leads to high amplitudes of MDCM charges and local positions
    res = minimize(
        mdcm_rmse, local_pos,
        method="L-BFGS-B",
        options={'disp': None, 'maxls': 20, 'iprint': -1, 'gtol': 1e-06, 
                 'eps': 1e-09, 'maxiter': 15000, 
                 'ftol': 1e-8, 'maxcor': 10, 'maxfun': 15000})
    print(res)

    # Recompute final RMSE each
    rmse = mdcm.get_rmse()
    print(rmse)
    srmse = mdcm.get_rmse_each(Nfiles)
    print(srmse)

    mdcm.write_cxyz_files()

    #  get the local charges array after optimization
    clcl_out = get_clcl(res.x, charges)

    # print(clcl_out)
    difference = np.sum((res.x - local_ref)**2)/local_pos.shape[0]
    print("charge RMSD:", difference)
    
    start_node = nodes_to_average[0]
    obj_name = f"pickles/{start_node}_clcl.obj"
    #  save as pickle
    filehandler = open(obj_name,"wb")
    pickle.dump(clcl_out, filehandler)

    # Not necessary but who knows when it become important to deallocate all 
    # global arrays
    mdcm.dealloc_all()

    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Scan and average for fMDCM')
    
    parser.add_argument('-f','--first', 
                        help='', required=False, default=False,
                       type=bool)
    
    parser.add_argument('-n','--nodes_to_avg', help='', required=True,
                       type=int, nargs="*")
    
    parser.add_argument('-l', '--local_pos', help='', default=None, type=str)
    
    args =parser.parse_args() 
    print(' '.join(f'{k}={v}\n' for k, v in vars(args).items()))
    
    if args.local_pos is not None:
        local = pd.read_pickle(args.local_pos)   
        optimize_mdcm(args.nodes_to_avg, first=args.first, local_pos=local)
    else:
        optimize_mdcm(args.nodes_to_avg, first=args.first)
    
    




