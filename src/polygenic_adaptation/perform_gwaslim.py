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
    parser.add_argument("-p", "--phenotypes", help="file containing the phenotypes of individuals")
    parser.add_argument("-g", "--genotypes", help="VCF file containing genotypes of individuals")
    parser.add_argument("-o", "--output", help="output file of estimated effect sizes and errors")
    smk = parser.parse_args()

    phenos = np.loadtxt(smk.phenotypes, delimiter=",")
    temp_genos = allel.read_vcf(smk.genotypes, fields=["GT", "S", "POS"])

    assert phenos.shape[0] == temp_genos["calldata/GT"].shape[1]

    # do some nonsense b/c SLiM's VCF output is messed up
    # even more nonsense to deal with SLiM not outputting both genotypes at fixed sites, kinda understandably tbh

    temp_freq_by_site = np.sum(np.sum(temp_genos["calldata/GT"], axis=-1), axis=-1)
    max_count = temp_genos["calldata/GT"].shape[1] * temp_genos["calldata/GT"].shape[2]
    max_locs = np.where(temp_freq_by_site == max_count)[0]
    fixed_calldata = np.copy(temp_genos["calldata/GT"])
    fixed_vardata = np.copy(temp_genos["variants/S"])
    for max_loc in max_locs[::-1]:
        fixed_calldata = np.concatenate(
            (
                fixed_calldata[: max_loc + 1, :, :],
                np.zeros((1, fixed_calldata.shape[1], fixed_calldata.shape[2]), dtype=int),
                fixed_calldata[max_loc + 1 :, :, :],
            )
        )
        if fixed_vardata[max_loc] == 0:
            # value doesn't matter, GWAS can't estimate anything anyway
            fixed_vardata = np.concatenate((fixed_vardata[: max_loc + 1], [0.01], fixed_vardata[max_loc + 1 :]))
        else:
            fixed_vardata = np.concatenate((fixed_vardata[: max_loc + 1], [0], fixed_vardata[max_loc + 1 :]))

    assert np.max(temp_genos["variants/POS"]) * 2 == fixed_calldata.shape[0]

    genos = fixed_calldata[fixed_vardata != 0, :, :]

    genotype_by_site = np.sum(genos, axis=-1)
    freq_by_site = np.sum(genotype_by_site, axis=-1)
    good_idxs = np.where((freq_by_site > 0) & (freq_by_site < max_count))[0]
    betas = np.zeros(genotype_by_site.shape[0])
    stderrs = np.zeros_like(betas)
    for site in np.arange(betas.shape[0]):
        if site in good_idxs:
            lr_result = linregress(genotype_by_site[site], phenos)
            betas[site] = lr_result.slope
            stderrs[site] = lr_result.stderr
    full_res = np.stack((betas, stderrs)).T
    np.savetxt(smk.output, full_res, header="1 row = 1 site, cols = (beta, stderr)")


if __name__ == "__main__":
    main()
