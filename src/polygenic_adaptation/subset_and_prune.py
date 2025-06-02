from __future__ import annotations

import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


def bh_correct(p_values, alpha, yekutieli=False):
    M = p_values.shape[0]
    p_values_sorted = np.sort(p_values.copy())
    bh_range = np.arange(1, M + 1)
    if yekutieli:
        alpha /= np.sum(1 / bh_range)
    small_ps = np.where(p_values_sorted <= bh_range * alpha / M)[0]
    if small_ps.shape[0] > 0:
        k_max = np.where(p_values_sorted <= bh_range * alpha / M)[0][-1]
    else:
        return 1, np.array([])
    p_k = np.sqrt(p_values_sorted[k_max] * p_values_sorted[k_max + 1])
    return p_k, np.where(p_values <= p_k)[0]


def find_peaks_greedy(ps, xs, chrs, min_height, window_size):
    ps_editable = np.copy(ps)
    idxs = []
    min_height_mask = ps_editable > min_height
    poss_to_real_idxs = np.arange(ps_editable.shape[0])[min_height_mask]
    poss_peaks = ps_editable[min_height_mask]
    poss_xs = xs[min_height_mask]
    poss_chrs = chrs[min_height_mask]
    if np.sum(min_height_mask) < 1:
        raise ValueError
    while np.max(poss_peaks) > min_height:
        max_idx = np.argmax(poss_peaks)
        surr_mask = (
            (poss_chrs == poss_chrs[max_idx])
            & (poss_xs < poss_xs[max_idx] + window_size)
            & (poss_xs > poss_xs[max_idx] - window_size)
        )
        poss_peaks[surr_mask] = 0
        idxs.append(poss_to_real_idxs[max_idx])
    return idxs


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "mode", nargs=1, type=str, help="currently 'LDetect' or 'greedy'"
    )
    parser.add_argument(
        "--height", nargs=1, type=float, help="min height to include peak"
    )
    parser.add_argument("--width", nargs=1, type=int, help="min distance between peaks")
    parser.add_argument("-i", "--input", nargs="*", help="input")
    parser.add_argument("-o", "--output", nargs="*", help="output")
    smk = parser.parse_args()

    assert smk.mode[0] == "LDetect" or smk.mode[0] == "greedy"
    alpha = 0.05
    trait_data_path = smk.input[0]
    shrunk_ests_path = smk.input[1]
    adna_snps_path = smk.input[2]
    adna_csv_path = smk.input[3]
    ld_blocks_path = smk.input[4] if smk.mode == "LDetect" else ""

    # subset the shrunk snps to adna snps
    with Path.open(adna_snps_path, "rb") as file:
        adna_data = pickle.load(file)

    sumstats_array = pd.read_csv(trait_data_path, sep="\t")

    ashres = pd.read_csv(shrunk_ests_path)
    sumstats_array[["ash_beta", "ash_se", "ash_p"]] = ashres[
        ["ash_beta", "ash_se", "ash_p"]
    ]

    p_bh, _ = bh_correct(sumstats_array["ash_p"].to_numpy(), alpha)

    paired_snps_alleles = zip(
        adna_data["all_rsid"],
        adna_data["all_ref_allele"],
        adna_data["all_alt_allele"],
        adna_data["all_chrom"],
        adna_data["all_loc_per_chrom"],
        strict=False,
    )
    filtered_sumstats_array = sumstats_array[
        sumstats_array[["SNP", "REF", "A2", "CHR", "POS"]]
        .apply(tuple, axis=1)
        .isin(paired_snps_alleles)
    ].reset_index()
    filtered_snps = filtered_sumstats_array["SNP"].to_numpy()
    assert filtered_snps.shape[0] == np.unique(filtered_snps).shape[0]

    if smk.mode[0] == "LDetect":
        # pick <= 1 per LD block
        ld_blocks = pd.read_csv(ld_blocks_path, delimiter="\t")
        ld_blocks.columns = ld_blocks.columns.str.strip()
        ld_blocks["chr"] = ld_blocks.chr.str[3:].astype(int)

        idxs = []
        for block in ld_blocks.itertuples(index=False):
            filtered_subset = filtered_sumstats_array[
                (filtered_sumstats_array["CHR"] == block[0])
                & (filtered_sumstats_array["POS"] >= block[1])
                & (filtered_sumstats_array["POS"] < block[2])
                & (filtered_sumstats_array["ash_p"] < p_bh)
            ]
            if filtered_subset.shape[0]:
                idxs.append(filtered_subset["ash_p"].idxmin())
    else:
        idxs = find_peaks_greedy(
            -np.log10(filtered_sumstats_array["ash_p"].to_numpy()),
            filtered_sumstats_array["POS"].to_numpy(),
            filtered_sumstats_array["CHR"].to_numpy(),
            float(smk.height[0]),
            int(smk.width[0]),
        )

    pruned_sumstats_array = filtered_sumstats_array.loc[idxs]

    pruned_sumstats_snps = pruned_sumstats_array["SNP"].to_numpy()
    pruned_adna_idxs = np.nonzero(
        pruned_sumstats_snps[:, None] == adna_data["all_rsid"]
    )[1]
    assert (
        np.intersect1d(
            adna_data["all_rsid"][pruned_adna_idxs], pruned_sumstats_snps
        ).shape[0]
        == pruned_adna_idxs.shape[0]
    )

    # compute heritability
    freqs = pruned_sumstats_array["EAF"].to_numpy()
    betas = pruned_sumstats_array["ash_beta"].to_numpy()

    h_squared = (freqs * (1 - freqs) * betas**2).sum()

    # save adna csv + betas and ses
    pruned_sumstats_array.to_csv(smk.output[0])
    np.savetxt(smk.output[1], np.concatenate(([h_squared], pruned_adna_idxs)))

    adna_csv = np.loadtxt(adna_csv_path, delimiter="\t")
    np.savetxt(
        smk.output[2],
        adna_csv[pruned_adna_idxs, :],
        delimiter="\t",
        fmt="%d",
        header="Each row = one replicate; each set of three columns = (sampling time, total samples, derived alleles)",
    )


if __name__ == "__main__":
    main()
