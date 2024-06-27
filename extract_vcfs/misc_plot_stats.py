from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AADR_VERSION = "v54.1"
EXTRACTED_PREFIX = f"extracted/GB_{AADR_VERSION}"
ANNO_FILE = pathlib.Path(f"{EXTRACTED_PREFIX}_inds.table")
COV_FILE = pathlib.Path(f"{EXTRACTED_PREFIX}_coverage.pdf")
SNP_FILE = pathlib.Path(f"{EXTRACTED_PREFIX}_1240k_hit.pdf")

TARGET_COVERAGE = "1240k_coverage"
SNPS_HIT = "SNPs_hit_1240k"


def plotCoverage(annoFrame):
    withCoverage = annoFrame.loc[annoFrame[TARGET_COVERAGE] != ".."]
    covs = np.array(withCoverage[TARGET_COVERAGE])
    logCovs = np.log10(covs.astype(float))
    (histY, histX) = np.histogram(logCovs, bins=20)
    plt.bar(0.5 * histX[1:] + 0.5 * histX[:-1], histY, width=histX[1] - histX[0])
    plt.xlabel("log10(coverage)")
    plt.tight_layout()
    plt.savefig(COV_FILE)
    plt.clf()


def plotSnpsHit(annoFrame):
    greatestHits = np.array(annoFrame[SNPS_HIT])
    logHits = np.log10(greatestHits.astype(float))
    (histY, histX) = np.histogram(logHits, bins=20)
    plt.bar(0.5 * histX[1:] + 0.5 * histX[:-1], histY, width=histX[1] - histX[0])
    plt.xlabel("log10(1240K hits)")
    plt.tight_layout()
    plt.savefig(SNP_FILE)
    plt.clf()


def main():
    # load anno file
    anno = pd.read_csv(ANNO_FILE, sep="\t", low_memory=False)

    # plot stats
    plotCoverage(anno)
    plotSnpsHit(anno)


if __name__ == "__main__":
    main()
