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
    parser.add_argument("--beta", type=float, help="beta (for use with --beta_mode constant)")
    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument("--omega", type=float, help="omega")
    parser.add_argument("--freq_init", type=float, help="constant initial freq (for use with --freq_mode constant")
    parser.add_argument("--freq_lb", type=float, help="initial freq lower bound (for use with --freq_mode uniform)")
    parser.add_argument("--freq_ub", type=float, help="initial freq upper bound (for use with --freq_mode uniform)")
    parser.add_argument(
        "--beta_file", type=str, help="path to beta file to sample from (for use with --beta_mode sample)"
    )
    parser.add_argument(
        "--freq_file", type=str, help="path to freq file to sample from (for use with --freq_mode sample)"
    )

    parser.add_argument("-n", type=int, help="num replicates")
    parser.add_argument("-o", "--output", nargs="*", help="output")
    smk = vars(parser.parse_args())

    betas = np.zeros(smk["n"])
    freqs = np.zeros(smk["n"])
    rng = np.random.default_rng(smk["seed"] + int(smk["omega"]))
    if smk["beta_mode"] == "constant":
        assert "beta" in smk
        betas += smk["beta"]
        signs = np.ones_like(betas)
        signs[signs.shape[0] // 2 :] *= -1
        betas = betas * signs  # rng.permutation(signs)
    elif smk["beta_mode"] == "sample":
        assert "beta_file" in smk
        base_betas = np.loadtxt(smk["beta_file"])
        betas = rng.choice(base_betas, smk["n"], replace=True)
    else:
        raise NotImplementedError

    if smk["freq_mode"] == "constant":
        assert "freq_init" in smk
        freqs += smk["freq_init"]
    elif smk["freq_mode"] == "uniform":
        assert "freq_lb" in smk
        assert "freq_ub" in smk
        assert smk["freq_lb"] <= smk["freq_ub"]
        freqs = rng.uniform(smk["freq_lb"], smk["freq_ub"], smk["n"])
    elif smk["freq_mode"] == "sample":
        assert "freq_file" in smk
        base_freqs = np.loadtxt(smk["freq_file"])
        freqs = rng.choice(base_freqs, smk["n"], replace=True)
    else:
        raise NotImplementedError

    np.savetxt(smk["output"][0], betas)
    np.savetxt(smk["output"][1], freqs)


if __name__ == "__main__":
    main()
