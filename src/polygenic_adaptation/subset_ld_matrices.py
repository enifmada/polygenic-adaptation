from __future__ import annotations

import pickle
from argparse import ArgumentParser
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.handlers import disable_signing
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
    except ValueError as ve:
        raise OSError("Corrupt file: %s" % (npz_file)) from ve

    # create df_R and return it
    # df_R = pd.DataFrame(R, index=df_ld_snps.index, columns=df_ld_snps.index)
    return R, df_ld_snps


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_fname", type=str, help=".txt file containing paths to files (on aws)")
    parser.add_argument("--bucket_name", type=str, help="name of aws bucket")
    parser.add_argument("--snp_file", type=str, help="path to file containing aDNA SNP info")
    parser.add_argument("--local_dir", type=str, help="path to local directory where files will be saved")
    parser.add_argument("--npz2_flag", action="store_true")

    smk = vars(parser.parse_args())

    s3 = boto3.resource("s3")
    s3.meta.client.meta.events.register("choose-signer.s3.*", disable_signing)

    bucket = s3.Bucket(smk["bucket_name"])

    with Path(smk["snp_file"]).open("rb") as f:
        all_adna_snp_data = pickle.load(f)

    all_rsid = all_adna_snp_data["all_rsid"]
    all_loc_per_chrom = all_adna_snp_data["all_loc_per_chrom"]

    local_dir = Path(smk["local_dir"])

    npz_suffix = ".npz2" if smk["npz2_flag"] else ".npz"

    input_npz = Path(smk["output_fname"].rpartition("_")[0] + npz_suffix).relative_to(local_dir)
    input_gz = input_npz.with_suffix(".gz")
    output_rsids = smk["output_fname"].rpartition("_")[0] + "_rsids.txt"

    local_npz = Path(smk["local_dir"]) / input_npz
    with local_npz.open("wb") as f:
        bucket.download_fileobj(str(input_npz), f)
    with local_npz.with_suffix(".gz").open("wb") as f:
        bucket.download_fileobj(str(input_gz), f)
    ld_m, ld_snps = load_ld_npz(local_npz.parent / local_npz.stem, npz_suffix)

    a, adna_idxs, snp_idxs = np.intersect1d(all_rsid, ld_snps["SNP"].to_numpy(), return_indices=True)

    if a.shape[0] > 0:
        snp_sort = np.argsort(snp_idxs)
        snp_locs = ld_snps["BP"].to_numpy()
        assert np.all(
            np.isclose(snp_locs[snp_idxs[snp_sort]], all_loc_per_chrom[adna_idxs[snp_sort]], atol=100, rtol=0)
        )

        reduced_matrix = ld_m[snp_idxs[snp_sort], snp_idxs[snp_sort]]
        np.save(smk["output_fname"], reduced_matrix)
        with Path(output_rsids).open("w") as f:
            for rsid in all_rsid[adna_idxs[snp_sort]]:
                f.write(rsid + "\n")
    else:
        np.save(smk["output_fname"], np.array([0]))
        with Path(output_rsids).open("w") as f:
            f.write("NO OVERLAPPING SNPS")
    local_npz.unlink()
    local_npz.with_suffix(".gz").unlink()


if __name__ == "__main__":
    main()
