from __future__ import annotations

import pathlib
import subprocess

import numpy as np
import pandas as pd

# where are the files?
AADR_VERSION = "v54.1.p1"
EXTRACTED_PREFIX = f"extracted/GB_{AADR_VERSION}"

# which genotypings do we have?
GENOTYPING = {
    'capture_only' : ['1240K', '1240k'],
    'capture_SG' : ['1240K', 'Shotgun', '1240k'],
}
# and which chromosomes?
CHROMOSOMES = [str(c) for c in np.arange(1,23)] + ['X', 'Y']

# the files
IND_FILES = {}
VCF_SKELETONS = {}
for geno in GENOTYPING:
    IND_FILES[geno] = pathlib.Path (f"{EXTRACTED_PREFIX}_{geno}_inds.table")
    VCF_SKELETONS[geno] = pathlib.Path (f"{EXTRACTED_PREFIX}_{geno}_c%s.vcf")
TMP_ID_FILE = "extracted/tmp.inds"

# just some binaries
VCFTOOLS_BINARY = "vcftools"




def extractCaptureOnly():

    # load annotation for all individuals
    annoCaptureSG = pd.read_csv (IND_FILES['capture_SG'], sep='\t')
    assert (set(annoCaptureSG['Data_source']).issubset(set(GENOTYPING['capture_SG'])))

    # get a mask for capture only ones
    captureOnlyMask = [(x in GENOTYPING['capture_only']) for x in annoCaptureSG['Data_source']]

    # write a table for the capture only individuals
    annoCaptureSG[captureOnlyMask].to_csv (IND_FILES['capture_only'], sep='\t', index=False)

    # get the IDs for the capture only individuals
    captureOnlyIDs = np.array(annoCaptureSG[captureOnlyMask]['Genetic_ID'])

    # write a file with IDs, cause vcftools needs that
    with pathlib.Path(TMP_ID_FILE).open("w") as ofs:
        ofs.write ("\n".join(captureOnlyIDs) + "\n")

    # and extract these individuals out of each VCF file
    for c in CHROMOSOMES:

        # get the files together
        thisInVcf = str(VCF_SKELETONS['capture_SG']) % c
        thisOutVcf = str(VCF_SKELETONS['capture_only']) % c

        # and put the bcf command together
        vcfCmd = f"{VCFTOOLS_BINARY} --vcf {thisInVcf} --keep {TMP_ID_FILE} --recode --stdout >{thisOutVcf}"

        # and run it
        subprocess.run([vcfCmd], shell=True, text=True, check=False)

    # and cleanup
    pathlib.Path(TMP_ID_FILE).unlink()


def main():
    extractCaptureOnly ()


if __name__ == "__main__":
    main()