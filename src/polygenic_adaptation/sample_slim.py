from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np


def get_params(fname):
    beta = 0
    omega = 0
    seed = 0
    beta_flag = False
    omega_flag = False
    seed_flag = False
    fparts = fname.split("_")
    for fpart in fparts[1:]:
        if fpart[0] == "b" and not beta_flag:
            beta = float(fpart[1:])
            beta_flag = True
        elif fpart[0] == "w" and not omega_flag:
            omega = float(fpart[1:])
            omega_flag = True
        elif fpart[0] == "s" and not seed_flag:
            seed = int(fpart[1:])
            seed_flag = True
    if omega > 0:
        return beta, omega, seed
    raise ValueError


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", nargs="*", help="input")
    parser.add_argument("-o", "--output", nargs="*", help="output")
    smk = parser.parse_args()

    rng = np.random.default_rng(5)
    _, omega, _ = get_params(smk.input[0])
    samples_per_timepoint = 50
    slim_array = np.loadtxt(smk.input[0], skiprows=1).T

    beta_path = Path(smk.input[0]).parent.parent / "betas"
    beta_fname = Path(smk.input[0]).name.rpartition("_")[0] + "_betas.txt"
    betas = np.loadtxt(beta_path / beta_fname)

    pop_genvar = np.sum(
        2 * betas[:, np.newaxis] ** 2 * slim_array[3:, :] * (1 - slim_array[3:, :]),
        axis=0,
    )
    final_array = np.zeros((slim_array.shape[0] - 3, slim_array.shape[1] * 3))

    final_array[:, ::3] = slim_array[0, :]
    final_array[:, 1::3] = samples_per_timepoint
    final_array[:, 2::3] = rng.binomial(samples_per_timepoint, slim_array[3:, :])

    p_samp = final_array[:, 2::3] / final_array[:, 1::3]
    samp_genvar = np.sum(2 * betas[:, np.newaxis] ** 2 * p_samp * (1 - p_samp), axis=0)

    np.savetxt(
        smk.output[0],
        final_array,
        delimiter="\t",
        fmt="%d",
        header="Each row = one replicate; each set of three columns = (sampling time; total samples; derived alleles)",
    )
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
    axs[0].plot(slim_array[3:, :].T)
    axs[1].plot(final_array[:, ::3].T, final_array[:, 2::3].T)
    axs[2].plot(slim_array[0, :], pop_genvar, label="pop")
    axs[2].plot(slim_array[0, :], samp_genvar, label="samp")
    axs[2].legend()
    axs[2].set_title(rf"$\omega^2=${omega**2}")
    fig.savefig(smk.output[1], format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
