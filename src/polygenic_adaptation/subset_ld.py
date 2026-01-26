from __future__ import annotations

import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse


def load_ld_npz(ld_prefix, suffix):
    # load the SNPs metadata
    gz_file = "%s.gz" % (ld_prefix)
    df_ld_snps = pd.read_table(gz_file, sep="\s+")
    df_ld_snps = df_ld_snps.rename(
        columns={"rsid": "SNP", "chromosome": "CHR", "position": "BP", "allele1": "A1", "allele2": "A2"},
        errors="ignore",
    )
    assert "SNP" in df_ld_snps.columns
    assert "CHR" in df_ld_snps.columns
    assert "BP" in df_ld_snps.columns
    assert "A1" in df_ld_snps.columns
    assert "A2" in df_ld_snps.columns
    df_ld_snps.index = (
        df_ld_snps["CHR"].astype(str)
        + "."
        + df_ld_snps["BP"].astype(str)
        + "."
        + df_ld_snps["A1"]
        + "."
        + df_ld_snps["A2"]
    )

    # load the LD matrix
    npz_file = ld_prefix.with_suffix(suffix)
    try:
        R = sparse.load_npz(npz_file).toarray()
        R += R.T
    except ValueError as npzve:
        raise OSError("Corrupt file: %s" % (npz_file)) from npzve

    # create df_R and return it
    # df_R = pd.DataFrame(R, index=df_ld_snps.index, columns=df_ld_snps.index)
    return R, df_ld_snps


def subset_ld(ld_file, output_loc, all_rsid, all_loc_per_chrom):
    input_path = output_loc.parent / Path(ld_file)
    input_basepath = input_path.parent / input_path.stem

    if not input_path.is_file():
        return 1

    output_npz = output_loc / (ld_file.rpartition(".")[0].rpartition("/")[-1] + "_subset.npz")
    output_txt = output_loc / (ld_file.rpartition(".")[0].rpartition("/")[-1] + "_rsids.txt")

    if output_npz.is_file():
        return 0

    try:
        ld_m, ld_snps = load_ld_npz(input_basepath, input_path.suffix)
    except ValueError:
        np.savez_compressed(output_npz, np.array([1]))
        with Path(output_txt).open("w") as f:
            f.write("WEIRD ZIP FILE")
        return 2

    a, adna_idxs, snp_idxs = np.intersect1d(all_rsid, ld_snps["SNP"].to_numpy(), return_indices=True)

    if a.shape[0] > 0:
        snp_sort = np.argsort(snp_idxs)
        snp_locs = ld_snps["BP"].to_numpy()
        assert np.all(
            np.isclose(snp_locs[snp_idxs[snp_sort]], all_loc_per_chrom[adna_idxs[snp_sort]], atol=100, rtol=0)
        )

        reduced_matrix = ld_m[np.ix_(snp_idxs[snp_sort], snp_idxs[snp_sort])]
        np.savez_compressed(output_npz, reduced_matrix)
        with Path(output_txt).open("w") as f:
            for rsid in all_rsid[adna_idxs[snp_sort]]:
                f.write(rsid + "\n")
    else:
        np.savez_compressed(output_npz, np.array([0]))
        with Path(output_txt).open("w") as f:
            f.write("NO OVERLAPPING SNPS")

    # print(f"saved {output_npz}!")
    return 0


def main():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output_dir", help="output directory")
    parser.add_argument("--ld_file_list", help="location of AWS file listing (used by download_ld.py)")
    parser.add_argument("--adna_data_file", help="location of aDNA data file")
    parser.add_argument("-nc", "--num_cores", type=int, help="number of cores")

    smk = vars(parser.parse_args())

    output_loc = Path(smk["output_dir"])

    with Path(smk["adna_data_file"]).open("rb") as f:
        all_adna_snp_data = pickle.load(f)

    all_rsid = all_adna_snp_data["all_rsid"]
    all_loc_per_chrom = all_adna_snp_data["all_loc_per_chrom"]

    with Path(smk["ld_file_list"]).open("r") as f:
        all_files = f.readlines()

    all_files = [f.strip() for f in all_files]
    all_files = [f for f in all_files if not f.endswith(".gz")]

    with Parallel(n_jobs=smk["num_cores"]) as parallel:
        parallel(delayed(subset_ld)(ldf, output_loc, all_rsid, all_loc_per_chrom) for ldf in all_files)


if __name__ == "__main__":
    main()
