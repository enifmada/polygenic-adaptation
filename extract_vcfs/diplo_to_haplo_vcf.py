from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def diploToHaploVcf(inputVcf):
    # we need the comments at the beginning of the vcf file
    with Path(inputVcf).open() as ifs:
        for line in ifs:
            # this allows us to get the vcf header into the dataframe
            if line.startswith("##"):
                pass
            else:
                assert line.startswith("#CHROM")
                # get the header
                columnNames = line.strip().split()

                # convert the rest with pandas
                vcfFrame = pd.read_csv(ifs, sep="\t", header=None, names=columnNames)
                break

    # which columns have the genotypes?
    # genotypes start after format column
    assert "FORMAT" in columnNames
    firstGenoColumn = columnNames.index("FORMAT") + 1
    # until the end

    # iterate over genotype columns
    for cIdx in np.arange(firstGenoColumn, vcfFrame.shape[1]):
        # get the name of the individual
        indName = columnNames[cIdx]

        # count the genotypes for this individuals
        thisGeno = vcfFrame.iloc[:, cIdx]
        thisCounter = collections.Counter(thisGeno)
        thisGenoSet = set(thisCounter.keys())

        # we only want valid genotypes (also, if missing, is missing in both)
        assert thisGenoSet.issubset({"./.", "0/0", "0/1", "1/0", "1/1"})

        # do we have any heterozygotes?
        if thisGenoSet.issubset({"./.", "0/0", "1/1"}):
            # no, only homozygotes
            # it is very likely that this is a pseudo-haploid individual
            # see if name of individual checks out
            assert ".DG" not in indName, indName

            # change it to be a proper haplotype
            vcfFrame.iloc[:, cIdx] = thisGeno.replace(
                to_replace=["./.", "0/0", "1/1"], value=[".", "0", "1"]
            )

        else:
            # we have heterozygotes, so this should be a regular diplotype
            # see if name of individual checks out
            assert ".DG" in indName, indName
            # don't change it

    # write it to stdout
    # header is already in stdout, so just write the rest
    vcfFrame.to_csv(sys.stdout, sep="\t", index=False)


def main():
    # need one parameter
    if len(sys.argv) != 2:
        sys.exit(-1)

    inputVcfFile = sys.argv[1]

    diploToHaploVcf(inputVcfFile)


if __name__ == "__main__":
    main()
