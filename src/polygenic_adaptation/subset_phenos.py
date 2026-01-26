from __future__ import annotations

import pickle
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def subset_pheno(input_file, output_file, adna_df):
    input_path = Path(input_file)

    if not input_path.is_file():
        return 1

    output_file = Path(output_file)

    if output_file.is_file():
        return 2

    pheno_file = pd.read_parquet(input_path)
    pheno_file[["chr", "loc", "ref", "alt"]] = pheno_file["variant"].str.split(":", expand=True)
    pheno_file = pheno_file.loc[pheno_file["chr"] != "X"]
    pheno_file = pheno_file.loc[~pheno_file["low_confidence_variant"]]
    pheno_file = pheno_file.reset_index(drop=True)
    pheno_file[["chr", "loc"]] = pheno_file[["chr", "loc"]].apply(pd.to_numeric)
    pheno_list = list(
        zip(
            pheno_file["chr"].values,
            pheno_file["loc"].values,
            pheno_file["ref"].values,
            pheno_file["alt"].values,
            strict=False,
        )
    )
    adna_list = list(
        zip(
            adna_df["all_chrom"].values,
            adna_df["all_loc_per_chrom"].values,
            adna_df["all_ref_allele"].values,
            adna_df["all_alt_allele"].values,
            strict=False,
        )
    )
    bool_mask = pd.Series(pheno_list).isin(adna_list)
    pheno_file_subset = pheno_file[bool_mask].reset_index(drop=True)
    red_pheno_list = list(
        zip(
            pheno_file_subset["chr"].values,
            pheno_file_subset["loc"].values,
            pheno_file_subset["ref"].values,
            pheno_file_subset["alt"].values,
            strict=False,
        )
    )
    opp_bool_mask = pd.Series(adna_list).isin(red_pheno_list)
    pheno_file_subset["rsid"] = adna_df["all_rsid"][opp_bool_mask].reset_index(drop=True)
    pheno_file_subset = pheno_file_subset.drop(columns=["variant", "low_confidence_variant"])
    pheno_file_subset.to_parquet(output_file, index=False)

    return 0


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="input ash file")
    parser.add_argument("-o", "--output", help="output subset file")
    parser.add_argument("--adna_data_file", help="location of aDNA data file")

    smk = vars(parser.parse_args())

    with Path(smk["adna_data_file"]).open("rb") as f:
        all_adna_snp_data = pickle.load(f)

    adna_df = pd.DataFrame(
        {
            k: all_adna_snp_data[k]
            for k in ["all_loc_per_chrom", "all_chrom", "all_rsid", "all_ref_allele", "all_alt_allele"]
        }
    )
    subset_pheno(smk["input"], smk["output"], adna_df)


if __name__ == "__main__":
    main()
