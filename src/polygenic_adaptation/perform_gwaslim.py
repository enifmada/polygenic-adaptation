from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import allel
import numpy as np
from scipy.stats import linregress


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-p", "--phenotypes", help="file containing the phenotypes of individuals"
    )
    parser.add_argument(
        "-g", "--genotypes", help="VCF file containing genotypes of individuals"
    )
    parser.add_argument(
        "-o", "--output", help="output file of estimated effect sizes and errors"
    )
    smk = parser.parse_args()

    phenos = np.loadtxt(smk.phenotypes, delimiter=",")
    temp_genos = allel.read_vcf(smk.genotypes)

    # do some nonsense b/c SLiM's VCF output is messed up
    freq_by_site = np.sum(np.sum(temp_genos["calldata/GT"], axis=-1), axis=-1)
    lowfreqidxs = np.where(
        freq_by_site[::2] < freq_by_site[1::2],
        np.arange(1000)[::2],
        np.arange(1000)[1::2],
    )
    genos = temp_genos["calldata/GT"][lowfreqidxs, :, :]
    genotype_by_site = np.sum(genos, axis=-1)
    betas = np.zeros(genotype_by_site.shape[0])
    stderrs = np.zeros_like(betas)
    for site in np.arange(betas.shape[0]):
        lr_result = linregress(genotype_by_site[site], phenos)
        betas[site] = lr_result.slope
        stderrs[site] = lr_result.stderr
    full_res = np.stack((betas, stderrs)).T
    np.savetxt(smk.output, full_res, header="1 row = 1 site, cols = (beta, stderr)")


if __name__ == "__main__":
    main()
