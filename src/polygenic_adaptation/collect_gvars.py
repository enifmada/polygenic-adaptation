from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", nargs="*", help="input")
    parser.add_argument("-o", "--output", nargs="*", help="output")
    smk = parser.parse_args()

    with Path.open(smk.output[0], "w") as file:
        for traitfile in smk.input:
            trait_name = traitfile.rpartition("/")[2].rpartition("ash")[0][:-1]
            h_squared = np.loadtxt(traitfile)[0]
            file.write(f"{trait_name}: {h_squared:.6f}")
            file.write("\n")

if __name__ == "__main__":
    main()