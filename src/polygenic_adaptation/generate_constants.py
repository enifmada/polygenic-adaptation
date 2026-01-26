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
    parser.add_argument("--vs", type=float, help="v_s")
    parser.add_argument("--freq_init", type=float, help="constant initial freq (for use with --freq_mode constant")
    parser.add_argument("--freq_lb", type=float, help="initial freq lower bound (for use with --freq_mode uniform)")
    parser.add_argument("--freq_ub", type=float, help="initial freq upper bound (for use with --freq_mode uniform)")
    parser.add_argument(
        "--beta_file", type=str, help="path to beta file to sample from (for use with --beta_mode sample)"
    )
    parser.add_argument(
        "--freq_file", type=str, help="path to freq file to sample from (for use with --freq_mode sample)"
    )
    parser.add_argument("--scale_factor", type=float, default=1, help="scale all betas by a constant amount")

    parser.add_argument("-n", type=int, help="num loci")
    parser.add_argument("-h2", type=float, help="heritability (for use with varmatched betas)")
    parser.add_argument(
        "-target_vP", type=float, default=1.0, help="desired phenotypic variance (for use with varmatched betas)"
    )
    parser.add_argument("-o", "--output", nargs="*", help="output")
    smk = vars(parser.parse_args())

    betas = np.zeros(smk["n"])
    freqs = np.zeros(smk["n"])
    rng = np.random.default_rng(smk["seed"] + int(smk["vs"]) * 12345)

    if smk["freq_mode"] == "constant":
        assert "freq_init" in smk
        assert smk["freq_init"] is not None
        freqs += smk["freq_init"]
    elif smk["freq_mode"] == "uniform":
        assert "freq_lb" in smk
        assert smk["freq_lb"] is not None
        assert "freq_ub" in smk
        assert smk["freq_ub"] is not None
        assert smk["freq_lb"] <= smk["freq_ub"]
        freqs = rng.uniform(smk["freq_lb"], smk["freq_ub"], smk["n"])
    elif smk["freq_mode"] == "sample":
        assert "freq_file" in smk
        assert smk["freq_file"] is not None
        base_freqs = np.loadtxt(smk["freq_file"])
        if "beta_file" in smk and smk["beta_file"] is not None:
            idx_array = rng.choice(np.arange(base_freqs.shape[0]), smk["n"], replace=True)
            freqs = base_freqs[idx_array]
        else:
            freqs = rng.choice(base_freqs, smk["n"], replace=True)
    elif smk["freq_mode"] == "matched":
        pass
    else:
        raise NotImplementedError

    if smk["beta_mode"] == "constant":
        assert "beta" in smk
        assert smk["beta"] is not None
        betas += smk["beta"]
        signs = np.ones_like(betas)
        signs[signs.shape[0] // 2 :] *= -1
        betas = betas * signs  # rng.permutation(signs)
    elif smk["beta_mode"] == "sample":
        assert "beta_file" in smk
        assert smk["beta_file"] is not None
        base_betas = np.loadtxt(smk["beta_file"])
        if "freq_file" in smk and smk["freq_file"] is not None:
            assert base_freqs.shape[0] == base_betas.shape[0]
            betas = base_betas[idx_array]
        else:
            betas = rng.choice(base_betas, smk["n"], replace=True)
    elif smk["beta_mode"] == "varmatched":
        assert "h2" in smk
        assert smk["h2"] is not None
        assert "target_vP" in smk
        assert smk["target_vP"] is not None
        v_g = smk["h2"] * smk["target_vP"]
        if "beta_file" in smk and smk["beta_file"] is not None:
            base_betas = np.loadtxt(smk["beta_file"])
            if "freq_file" in smk and smk["freq_file"] is not None:
                assert base_freqs.shape[0] == base_betas.shape[0]
                betas = base_betas[idx_array]
            else:
                betas = rng.choice(base_betas, smk["n"], replace=True)
            emp_vg = np.sum(2 * betas**2 * freqs * (1 - freqs))
            betas *= np.sqrt(v_g / emp_vg)
        else:
            # if no beta file, beta**2 is constant
            # nope!! make additional flag for this
            beta_val = np.sqrt(v_g / (2 * np.sum(freqs * (1 - freqs))))
            betas += beta_val
            signs = np.ones_like(betas)
            signs[signs.shape[0] // 2 :] *= -1
            betas = betas * signs
        assert np.isclose(v_g, 2 * np.sum(betas**2 * freqs * (1 - freqs)))
    elif smk["beta_mode"] == "matched":
        pass
    else:
        raise NotImplementedError

    if smk["beta_mode"] != "matched":
        betas *= smk["scale_factor"]
        np.savetxt(smk["output"][0], betas)
    if smk["freq_mode"] != "matched":
        np.savetxt(smk["output"][1], freqs)


if __name__ == "__main__":
    main()
