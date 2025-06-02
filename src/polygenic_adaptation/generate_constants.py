from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np


def main():
    parser = ArgumentParser()
    parser.add_argument("--beta_mode", type=str, help="beta mode")
    parser.add_argument("--freq_mode", type=str, help="freq mode")
    parser.add_argument("--beta", type=float, help="beta")
    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument("--omega", type=float, help="omega")
    parser.add_argument("--freq", type=float, help="freq")
    parser.add_argument("-n", type=int, help="num replicates")
    parser.add_argument("-o", "--output", nargs="*", help="output")
    smk = parser.parse_args()

    betas = np.zeros(smk.n)
    freqs = np.zeros(smk.n)
    np.random.default_rng(smk.seed + int(smk.omega))
    if smk.beta_mode == "constant":
        betas += smk.beta
        signs = np.ones_like(betas)
        signs[signs.shape[0] // 2 :] *= -1
        betas = betas * signs  # rng.permutation(signs)
    else:
        raise NotImplementedError

    if smk.freq_mode == "constant":
        freqs += smk.freq

    np.savetxt(smk.output[0], betas)
    np.savetxt(smk.output[1], freqs)


if __name__ == "__main__":
    main()
