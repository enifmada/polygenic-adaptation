from __future__ import annotations

import collections
import pathlib
import subprocess

import pandas as pd

AADR_VERSION = "v54.1.p1"
# we take the 1240K version here
AADR_ROOT = pathlib.Path(f"AADR/{AADR_VERSION}_1240K/{AADR_VERSION}_1240K_public")
AADR_SNP = pathlib.Path(f"{AADR_ROOT}.snp")
GENETIC_ID = "Genetic_ID"

# file specifying individuals to extract
EXTRACTED_PREFIX = f"extracted/GB_{AADR_VERSION}_capture_SG"
INDIVIDUALS_FILE = pathlib.Path(f"{EXTRACTED_PREFIX}_inds.table")

# external scripts
EIGENSTRAT_CONVERSION_SCRIPT = pathlib.Path("gdc/eigenstrat2vcf.py")
CORRECT_HAPLO_ENCODING_SCRIPT = pathlib.Path("diplo_to_haplo_vcf.py")


def extractTimeSeries():
    # load the individuals and make an individuals file for conversion
    conversionIndFile = pathlib.Path(f"{EXTRACTED_PREFIX}_inds.conv")
    individualsFrame = pd.read_csv(INDIVIDUALS_FILE, sep="\t")
    with pathlib.Path(conversionIndFile).open("w", encoding="locale") as ofs:
        for thisInd in individualsFrame[GENETIC_ID]:
            ofs.write(f"{thisInd}\n")

    # load all the snps
    snpFrame = pd.read_csv(AADR_SNP, sep=r"\s+")
    chromHist = collections.Counter(snpFrame.iloc[:, 1])
    # make sure all chromosomes accounted for
    assert len(chromHist.keys()) <= 24, len(chromHist.keys())
    assert min(chromHist.keys()) == 1, min(chromHist.keys())
    assert max(chromHist.keys()) == 24, max(chromHist.keys())

    # one vcf-file for each chromosome
    for c in chromHist:
        chromName = str(c)
        # 23 is X
        if c == 23:
            chromName = "X"
        # 24 is Y
        elif c == 24:
            chromName = "Y"
        else:
            pass

        # make a snpfile for eigenstrat
        conversionSnpFile = pathlib.Path(f"{EXTRACTED_PREFIX}_c{chromName}.snps")
        thisSnpFrame = snpFrame.loc[snpFrame.iloc[:, 1] == c]
        with pathlib.Path(conversionSnpFile).open("w", encoding="locale") as ofs:
            for thisSnp in thisSnpFrame.iloc[:, 0]:
                ofs.write(f"{thisSnp}\n")

        # prepare the output file
        outputDiploVCF = pathlib.Path(f"{EXTRACTED_PREFIX}_c{chromName}.diplo_vcf")

        # put eigentstrat command together
        stratCmd = f"python {EIGENSTRAT_CONVERSION_SCRIPT} -r {AADR_ROOT} -i {conversionIndFile} -s {conversionSnpFile} >{outputDiploVCF}"

        # extract it
        subprocess.run([stratCmd], shell=True, text=True, check=False)

        # clean up
        pathlib.Path(conversionSnpFile).unlink()

        # and make it a proper vcf with haploid calls encoded correctly
        outputVCF = pathlib.Path(f"{EXTRACTED_PREFIX}_c{chromName}.vcf")

        haploCmd = (
            f"python {CORRECT_HAPLO_ENCODING_SCRIPT} {outputDiploVCF} > {outputVCF}"
        )

        # convert it to pseudo-haploid
        subprocess.run([haploCmd], shell=True, text=True, check=False)

        # clean up
        pathlib.Path(outputDiploVCF).unlink()


def main():
    extractTimeSeries()


if __name__ == "__main__":
    main()
