from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

AADR_VERSION = "v54.1.p1"
EXTRACTED_PREFIX = f"extracted/GB_{AADR_VERSION}_capture_SG"
INPUT_TABLE = pathlib.Path(f"{EXTRACTED_PREFIX}_pre_pca_inds.table")
OUTPUT_TABLE = pathlib.Path(f"{EXTRACTED_PREFIX}_inds.table")
# # these ones look like outliers in the PCA plot
# GENETIC_IDS_TO_REMOVE = ['I11570']
# for now no removal
GENETIC_IDS_TO_REMOVE = []


def removeIds():
    # load old anno
    anno = pd.read_csv(INPUT_TABLE, sep="\t", low_memory=False)

    # remove some genetic ids
    removeMask = np.array([x in GENETIC_IDS_TO_REMOVE for x in anno["Genetic_ID"]])
    newAnno = anno.loc[~removeMask]

    # store new anno
    newAnno.to_csv(OUTPUT_TABLE, sep="\t", index=False)


def main():
    removeIds()


if __name__ == "__main__":
    main()
