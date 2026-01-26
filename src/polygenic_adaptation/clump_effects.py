from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


def clump_peaks(ps, xs, r2_matrix, min_r2, min_height, window_size_kb):
    ps_editable = np.copy(ps)
    idxs = []
    min_height_mask = ps_editable < min_height
    if min_height_mask.sum() < 1:
        return idxs
    poss_to_real_idxs = np.arange(ps_editable.shape[0])[min_height_mask]
    poss_peaks = ps_editable[min_height_mask]
    poss_xs = xs[min_height_mask]
    poss_r2 = r2_matrix[min_height_mask, :]
    while np.min(poss_peaks) < min_height:
        max_idx = np.argmin(poss_peaks)
        in_ld_locs = xs[poss_r2[max_idx] > min_r2]
        in_ld_min = np.min(in_ld_locs)
        in_ld_max = np.max(in_ld_locs)
        mask_ub = min(in_ld_max - poss_xs[max_idx], 1000 * window_size_kb)
        mask_lb = min(poss_xs[max_idx] - in_ld_min, 1000 * window_size_kb)
        surr_mask = (poss_xs <= poss_xs[max_idx] + mask_ub) & (poss_xs >= poss_xs[max_idx] - mask_lb)
        poss_peaks[surr_mask] = np.inf
        idxs.append(poss_to_real_idxs[max_idx])
    return idxs


def clump_block(trait_df, ld_fpath, min_r2, min_height, window_size_kb):
    # print(f"we clumpin {ld_fpath}")
    ld_path = Path(ld_fpath)
    chr_info, block_min, block_max, _ = ld_path.stem.split("_")
    cur_chr = int(chr_info[3:])
    block_min = int(block_min)
    block_max = int(block_max)
    ld_matrix = np.load(ld_path)["arr_0"]
    rsids = np.loadtxt(ld_path.parent / (ld_path.stem.rpartition("_")[0] + "_rsids.txt"), dtype=str)
    if rsids.ndim == 0:
        rsids = rsids[np.newaxis]
    if rsids[0][:2] == "NO" or rsids[0][:2] == "WE":
        return []
    trait_subset = trait_df[
        (trait_df["chr"] == cur_chr) & (trait_df["loc"] > block_min) & (trait_df["loc"] < block_max)
    ]
    trait_rsid = trait_subset["rsid"].to_numpy()
    rsid_set = set(rsids.tolist())
    trait_rsid_set = set(trait_rsid.tolist())

    # intr, rsid_mask, trait_mask = np.intersect1d(rsids, trait_df['rsid'].to_numpy(), assume_unique=True, return_indices=True)

    intr = np.array(list(rsid_set.intersection(trait_rsid_set)))
    rsid_mask = np.isin(rsids, intr)
    red_df = trait_subset[trait_subset["rsid"].isin(intr)]

    if red_df.shape[0] <= 0:
        return []
    # idxs_sorted = np.argsort(trait_mask)

    # red_df = trait_df.loc[trait_mask]
    rsids_red = rsids[rsid_mask]
    ld_red = ld_matrix[np.ix_(rsid_mask, rsid_mask)]
    afs = red_df["minor_AF"].to_numpy()
    af_pq_matrix = afs * (1 - afs) * afs[:, np.newaxis] * (1 - afs)[:, np.newaxis]
    neg_matrix = np.minimum(afs * afs[:, np.newaxis], (1 - afs) * (1 - afs)[:, np.newaxis])
    pos_matrix = np.minimum(afs * (1 - afs)[:, np.newaxis], (1 - afs) * afs[:, np.newaxis])
    neg_matrix[ld_red > 0] = pos_matrix[ld_red > 0]
    ld_proper = neg_matrix * ld_red
    r2_matrix = ld_proper**2 / af_pq_matrix
    test_idxs_set = clump_peaks(
        red_df["ash_p"].to_numpy(), red_df["loc"].to_numpy(), r2_matrix, min_r2, min_height, window_size_kb
    )
    return rsids_red[test_idxs_set].tolist()


def clump_trait(trait_fpath, clump_dir, clump_suffix, all_ld_list, min_r2=0.5, min_height=1e-3, window_size_kb=250):
    trait_path = Path(trait_fpath)
    pheno_ID = trait_path.stem.rpartition("_")[0]
    parquet_path = Path(clump_dir) / f"{pheno_ID}{clump_suffix}"
    if parquet_path.is_file():
        return 1

    trait_df = pd.read_parquet(trait_fpath)
    rsids = []
    for ld_fpath in all_ld_list:
        rsids.extend(clump_block(trait_df, ld_fpath, min_r2, min_height, window_size_kb))

    rsids = list(set(rsids))
    trait_subset = trait_df[trait_df["rsid"].isin(rsids)]

    # adna_mask = np.isin(adna_rsids, rsids)
    # adna_subset = adna_data[adna_mask]

    trait_subset.to_parquet(parquet_path)
    return 0


def main():
    parser = ArgumentParser()
    parser.add_argument("--ld_dir", help="LD files directory")
    parser.add_argument("--pheno_file", help="phenotype file")
    parser.add_argument("--clump_dir", help="clumped files directory")
    parser.add_argument("--ld_list_file", help="location of ld file listing")
    parser.add_argument("--clump_suffix", help="suffix of the clumped files")
    parser.add_argument("--min_r2", type=float, default=0.5, help="minimum r^2 value to be in a clump")
    parser.add_argument("--min_height", type=float, default=1e-3, help="minimum p value to be center SNP of a clump")
    parser.add_argument("--window_size_kb", type=int, default=250, help="maximum size of clumps (IN KB)")

    smk = vars(parser.parse_args())
    with Path(smk["ld_list_file"]).open("r") as f:
        all_ld_files = f.readlines()

    all_ld_files = [f.strip() for f in all_ld_files]
    all_ld_files = [f for f in all_ld_files if ".gz" not in f]
    all_ld_files = [smk["ld_dir"] + "/" + f.split("/")[1] for f in all_ld_files]
    all_ld_files = [f.rpartition(".")[0] + "_subset.npz" for f in all_ld_files]

    clump_trait(
        smk["pheno_file"],
        smk["clump_dir"],
        smk["clump_suffix"],
        all_ld_files,
        smk["min_r2"],
        smk["min_height"],
        smk["window_size_kb"],
    )


if __name__ == "__main__":
    main()
