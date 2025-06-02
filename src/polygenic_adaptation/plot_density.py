from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", nargs="*", help="input")
    parser.add_argument("-o", "--output", nargs="*", help="output")
    smk = parser.parse_args()

    init_array = np.loadtxt(smk.input[0])
    pis = init_array[: init_array.shape[0] // 2]
    sigmas = init_array[init_array.shape[0] // 2 :]
    large_density = norm(loc=0, scale=sigmas[np.nonzero(pis)[0][-2]])
    small_density = norm(loc=0, scale=sigmas[np.nonzero(pis)[0][3]])
    large_xspace = np.linspace(large_density.ppf(0.01), large_density.ppf(0.99), 1000)
    small_xspace = np.linspace(small_density.ppf(0.01), small_density.ppf(0.99), 200)
    large_density = np.zeros(large_xspace.shape[0] - 1)
    small_density = np.zeros(small_xspace.shape[0] - 1)

    for p_i in np.arange(1, pis.shape[0]):
        large_density += pis[p_i] * np.diff(
            norm(loc=0, scale=sigmas[p_i]).cdf(large_xspace)
        )
        small_density += pis[p_i] * np.diff(
            norm(loc=0, scale=sigmas[p_i]).cdf(small_xspace)
        )
    small_density[small_density.shape[0] // 2] += pis[0]
    large_density[large_density.shape[0] // 2] += pis[0]

    plot_space_large = large_xspace[:-1] + np.diff(large_xspace) / 2
    plot_space_small = small_xspace[:-1] + np.diff(small_xspace) / 2

    plot_title = smk.input[0].rpartition("/")[-1].rpartition("ash")[0][:-1]
    fig, axs = plt.subplots(1, 1, figsize=(10, 5), layout="constrained")
    axs.fill_between(plot_space_small, small_density, color="gray", alpha=0.5)
    fig.suptitle(f"{plot_title} density plot")
    fig.savefig(smk.output[0], format="pdf", bbox_inches="tight")

    fig, axs = plt.subplots(1, 1, figsize=(10, 5), layout="constrained")
    axs.fill_between(plot_space_large, np.log(large_density), color="gray", alpha=0.5)
    fig.suptitle(f"{plot_title} hatplot (log density)")
    fig.savefig(smk.output[1], format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
