from __future__ import annotations

import argparse
import sys
from pathlib import Path

import allel
import numpy as np
import pandas as pd

# probably need to change this once we convert to an actual module like last time
sys.path.append(
    str(Path(__file__).resolve().parent.parent / "src" / "polygenic_adaptation")
)
from polyutil import vcf_to_useful_format


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_vcf", type=str, help="path to input VCF")
    parser.add_argument("input_table", type=str, help="path to input .table")
    parser.add_argument("output_vcf", type=str, help="path to output filtered VCF")
    parser.add_argument(
        "-maf",
        "--min_allele_freq",
        type=float,
        default=0.05,
        help="filters out replicates with mean minor allele frequency < MAF",
    )
    parser.add_argument(
        "--min_sample_density",
        type=float,
        default=0.1,
        help="filters out replicates with fewer than (min_sample_density * max_samples) total samples",
    )

    args = parser.parse_args()

    output_bigstr = ""
    # we need the comments at the beginning of the vcf file
    with Path(args.input_vcf).open() as ifs:
        for line in ifs:
            # this allows us to get the vcf header into the dataframe
            if line.startswith("##"):
                output_bigstr += line.strip() + "\n"
            else:
                assert line.startswith("#CHROM")
                # get the header
                columnNames = line.strip().split()
                # convert the rest with pandas
                vcfFrame = pd.read_csv(ifs, sep="\t", header=None, names=columnNames)
                break

    # independently, get a formatted CSV to figure out how to filter
    vcf_dates = pd.read_csv(
        args.input_table, usecols=["Genetic_ID", "Date_mean"], sep="\t"
    ).to_numpy()
    vcf_file = allel.read_vcf(args.input_vcf)
    max_samples = (
        vcf_file["calldata/GT"].shape[1]
        + np.any(vcf_file["calldata/GT"][:, :, 1] >= 0, axis=0).sum()
    )

    conv_csv = vcf_to_useful_format(vcf_file, vcf_dates, force="haploid")

    assert vcfFrame.shape[0] == conv_csv.shape[0]

    fdata = conv_csv[:, 2::3]
    nsamples = conv_csv[:, 1::3]
    total_fd = np.sum(fdata, axis=1)
    total_ns = np.sum(nsamples, axis=1)

    min_fd = np.minimum(total_fd, total_ns - total_fd)
    MAF_filter_mask = min_fd > total_ns * args.min_allele_freq

    num_samples_mask = np.sum(nsamples != 0, axis=1) > 1
    anc_samples_mask = np.sum(nsamples, axis=1) > args.min_sample_density * max_samples
    combo_mask = num_samples_mask & MAF_filter_mask & anc_samples_mask

    vcfFrame = vcfFrame[combo_mask]
    # write the csv to the output string
    output_bigstr += vcfFrame.to_csv(sep="\t", index=False)
    with Path(args.output_vcf).open("w") as file:
        file.write(output_bigstr)


if __name__ == "__main__":
    main()
