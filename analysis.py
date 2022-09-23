
import matplotlib.pyplot as plt

import os
import networkx as nx
import numpy as np
import pandas as pd



from scipy.stats import pearsonr, spearmanr


bohr_to_a = 0.529177


def scale_min_max(data, x):
    # print(data)
    # print(x)
    return (x - data.min())/(data.max() - data.min())

def scale_Z(data, x):
    # print(data)
    # print(x)
    return (x - data.mean())/(data.std())


def get_dist_matrix(atoms):
    #https://www.kaggle.com/code/rio114/coulomb-interaction-speed-up/notebook
    num_atoms = len(atoms)
    loc_tile = np.tile(atoms.T, (num_atoms,1,1))
    dist_mat = np.sqrt((loc_tile - loc_tile.T)**2).sum(axis=1)
    return dist_mat

def dihedral3(p):
    b = p[:-1] - p[1:]
    b[0] *= -1
    v = np.array( [np.cross(v,b[1]) for v in [b[0], b[2]] ] )
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1,1)
    return np.degrees(np.arccos( v[0].dot(v[1]) ))


def angle(p):
    ba = p[0] - p[1]
    bc = p[2] - p[1]
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    a3 = np.degrees(np.arccos(cosine_angle))
    return a3

def load_nc(path):
    #  load nuclear coordinates
    nc_lines = open(path).readlines()[6:12]
    ncs = np.array([[float(y)*bohr_to_a for y in x.split()[2:]] for x in nc_lines])
    return ncs


def get_edge_weights(node, G):
    edges = G.edges(node)
    nodes = []
    weights = []
    for e in edges:
        nodes.append(e[1])
        weights.append(G.get_edge_data(*e)["weight"])
    return nodes, weights


RMSD_to_ref = pd.read_pickle("/home/unibas/boittier/fdcm_python/RMSD_to_ref.obj")


def compare_distances(i, lcs_df, uptri_df, G):
    comp_lcs = lcs_df.loc[i]
    comp_nc_uptri = uptri_df.loc[i]

    neighbour_lcs = []
    lcs_distances = []
    nc_uptri_distances = []

    edge_nodes, edge_weights = get_edge_weights(i, G)

    #  scan through edges
    for n, w in zip(edge_nodes, edge_weights):
        if n in list(lcs_df.index):
            tmp_lcs = lcs_df.loc[n]
            neighbour_lcs.append(tmp_lcs)

            lc_dist = np.sqrt(np.mean((comp_lcs - tmp_lcs) ** 2))
            lcs_distances.append(lc_dist)
            nc_uptri_dist = np.sqrt(np.mean((comp_nc_uptri - uptri_df.loc[n]) ** 2))
            nc_uptri_distances.append(nc_uptri_dist)

    # print(f"mean distance: {np.mean(lcs_distances)}")

    # if plot:
    #     plt.scatter(nc_uptri_distances, lcs_distances)
    #     plt.title("{}\n{}\n{}".format(i, pearsonr(nc_uptri_distances, lcs_distances),
    #                                   spearmanr(nc_uptri_distances, lcs_distances)))
    #     plt.xlabel("NC RMSD")
    #     plt.ylabel("LC RMSD")
    #     plt.show()
    #     plt.clf()

    return pearsonr(nc_uptri_distances, lcs_distances)


def analysis(df_name, G):
    ref_pos = load_nc("/home/unibas/boittier/MDCM/examples/multi-conformer/ref/scan0.p.cube")
    ref_dm = get_dist_matrix(ref_pos)

    outdir = f"/data/unibas/boittier/{df_name}/output"
    pkl_path = f"/data/unibas/boittier/{df_name}/pickles"

    outfiles = os.listdir(outdir)
    print("# outfiles:", len(outfiles))

    initial_rmsds = {}
    final_rmsds = {}
    charge_rmsds = {}
    lcs = {}
    ncs = {}
    distM = {}
    ref_rmsd = {}
    flat_DMs = {}
    uptri_DMs = {}
    lcs_scaled = {}

    iu1 = np.triu_indices(6)

    for outfile in outfiles:
        index = int(outfile.split(".")[0])
        lines = open(os.path.join(outdir, outfile)).readlines()
        try:
            initial_rmsd = 0.0  # float(lines[5])
            final_rmsd = float([x for x in lines if x.__contains__("fun:")][0].split()[1])
            charge_rmsd = float([x for x in lines if x.__contains__("charge RMSD:")][0].split()[2])

            initial_rmsds[index] = initial_rmsd
            final_rmsds[index] = final_rmsd
            charge_rmsds[index] = charge_rmsd

            obj = f"{index}_clcl.obj"
            pkl = pd.read_pickle(os.path.join(pkl_path, obj))
            local = pkl[np.mod(np.arange(pkl.size) + 1, 4) != 0]
            lcs[index] = local
            lcs_scaled[index] = scale_min_max(local, local)

            _path_ = f"/data/unibas/boittier/graphscan/methanol/t3/p{index}.p.cube"
            nc = load_nc(_path_)
            ncs[index] = nc
            dm = get_dist_matrix(nc)
            distM[index] = dm

            flat_dm = dm.flatten()
            flat_DMs[index] = flat_dm

            # reduce to only the upper triangle, no diagonals (zeros)
            uptri = dm[iu1]
            uptri_dm = uptri[uptri != 0]
            uptri_dm = scale_min_max(uptri_dm, uptri_dm)
            uptri_DMs[index] = uptri_dm

            ref_rmsd[index] = np.linalg.norm(dm - ref_dm)

        except Exception as e:
            print(outfile, e)

    df = pd.DataFrame({"rmsd_i": initial_rmsds,
                       "rmsd_f": final_rmsds,
                       "chg_rmsd": charge_rmsds,
                       "ref_rmsd": ref_rmsd,
                       "lcs": lcs,
                       "ncs": ncs,
                       "distM": distM,
                       "flatDM": flat_DMs,
                       "uptriDM": uptri_DMs})

    df["rmsd_dif"] = df["rmsd_i"] - df["rmsd_f"]

    df.to_pickle(f"data/{df_name}.obj")

    lcs_df = pd.DataFrame(lcs).T
    rename = {i: f"q{i // 3}ax{i % 3}" for i in range(30)}
    lcs_df.rename(columns=rename, inplace=True)

    uptri_df = pd.DataFrame(uptri_DMs).T

    pearsons_r = []
    pearsons_p = []

    for i in list(G.nodes()):
        try:
            r, p = compare_distances(i, lcs_df, uptri_df)
            pearsons_r.append(r)
            pearsons_p.append(p)

        except Exception as e:
            # print(i, e)
            pass


