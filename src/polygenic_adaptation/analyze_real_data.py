from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", nargs="*", help="input")
    parser.add_argument("-o", "--output", nargs="*", help="output")

    smk = parser.parse_args()

    # MIN_BETA_VALUE = 1e-5
    direc_unif_estimates = []
    stab_unif_estimates = []
    trait_names = []
    num_inputs = len(smk.input)
    for grid_i in tqdm(range(num_inputs // 2)):
        trait_name = Path(smk.input[grid_i]).name.rpartition("_")[0].rpartition("_")[0]
        assert trait_name in Path(smk.input[grid_i + num_inputs // 2]).name
        trait_names.append(trait_name)
        grid = np.loadtxt(smk.input[grid_i])
        sumstats = pd.read_csv(smk.input[grid_i + num_inputs // 2])
        betas = sumstats["ash_beta"].to_numpy()
        raw_grid = grid[0, :]
        dll_unif_vals = grid[1::2, :]
        sll_unif_vals = grid[2::2, :]

        max_signed_beta = np.max(np.abs(betas))
        expanded_direc_x = np.linspace(
            raw_grid[0] / (2 * max_signed_beta),
            raw_grid[-1] / (2 * max_signed_beta),
            10000,
        )
        direc_s2l_raw_1 = raw_grid[1] / (2 * max_signed_beta)
        direc_s2l_raw_2 = raw_grid[-2] / (2 * max_signed_beta)
        expanded_stab_x = np.linspace(
            raw_grid[0] / (max_signed_beta**2 / 2),
            raw_grid[-1] / (max_signed_beta**2 / 2),
            10000,
        )
        stab_s2l_raw_1 = raw_grid[1] / (max_signed_beta**2 / 2)
        stab_s2l_raw_2 = raw_grid[-2] / (max_signed_beta**2 / 2)
        summed_unif_dlls = np.zeros_like(expanded_direc_x)
        summed_unif_slls = np.zeros_like(expanded_stab_x)
        all_dll_unif_ests = np.zeros((dll_unif_vals.shape[0], summed_unif_dlls.shape[0]))
        all_sll_unif_ests = np.zeros((dll_unif_vals.shape[0], summed_unif_dlls.shape[0]))
        for loc in range(dll_unif_vals.shape[0]):
            # *2 b/c conversion from s2 = s to s1 = s
            sdz_est_grid = raw_grid / (2 * betas[loc])

            # /2 b/c I think that's the right coefficient in the actual equations?
            s_est_grid = raw_grid / (betas[loc] ** 2 / 2)
            sll_unif_spline = CubicSpline(s_est_grid, sll_unif_vals[loc, :])
            if betas[loc] >= 0:
                dll_unif_spline = CubicSpline(sdz_est_grid, dll_unif_vals[loc, :])
            else:
                dll_unif_spline = CubicSpline(sdz_est_grid[::-1], dll_unif_vals[loc, ::-1])

            dll_unif_ests = dll_unif_spline(expanded_direc_x)
            all_dll_unif_ests[loc, :] = dll_unif_ests
            summed_unif_dlls += dll_unif_ests

            sll_unif_ests = sll_unif_spline(expanded_stab_x)
            all_sll_unif_ests[loc, :] = sll_unif_ests
            summed_unif_slls += sll_unif_ests

        fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
        for loc in range(dll_unif_vals.shape[0]):
            axs[0].plot(expanded_direc_x, all_dll_unif_ests[loc, :])
            axs[1].plot(expanded_stab_x, all_sll_unif_ests[loc, :])
        axs[0].plot(
            expanded_direc_x,
            summed_unif_dlls / all_dll_unif_ests.shape[0],
            color="k",
            lw=2,
        )
        axs[1].plot(
            expanded_stab_x,
            summed_unif_slls / all_dll_unif_ests.shape[0],
            color="k",
            lw=2,
        )
        axs[0].axvline(direc_s2l_raw_1, color="k", ls="--")
        axs[0].axvline(direc_s2l_raw_2, color="k", ls="--")
        axs[1].axvline(stab_s2l_raw_1, color="k", ls="--")
        axs[1].axvline(stab_s2l_raw_2, color="k", ls="--")
        axs[0].set_title("Unif Direc")
        axs[1].set_title("Unif Stab")
        fig.suptitle(f"{grid_i}")
        fig.savefig(
            Path(smk.output[1]).parent / f"{Path(smk.input[grid_i]).stem}_all_pchip_ests.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(fig)
        np.max(summed_unif_dlls)
        dll_unif_argmax = expanded_direc_x[np.argmax(summed_unif_dlls)]

        direc_unif_estimates.append(dll_unif_argmax)

        np.max(summed_unif_slls)
        sll_unif_argmax = expanded_stab_x[np.argmax(summed_unif_slls)]

        stab_unif_estimates.append(sll_unif_argmax)

    trait_names = np.array(trait_names)
    direc_unif_estimates = np.array(direc_unif_estimates)
    stab_unif_estimates = np.array(stab_unif_estimates)
    direc_argsort = np.argsort(direc_unif_estimates)
    stab_argsort = np.argsort(stab_unif_estimates)
    fig1, axs1 = plt.subplots(1, 1, figsize=(15, 15), layout="constrained")
    axs1.plot(
        np.arange(len(direc_unif_estimates)) + 1,
        direc_unif_estimates[direc_argsort],
        "b*",
    )
    axs1.set_xticks(
        np.arange(len(direc_unif_estimates)) + 1,
        labels=trait_names[direc_argsort],
        rotation=90,
    )
    fig1.savefig(smk.output[0], format="pdf", bbox_inches="tight")
    plt.close(fig1)

    fig2, axs2 = plt.subplots(1, 1, figsize=(15, 15), layout="constrained")
    axs2.plot(np.arange(len(stab_unif_estimates)) + 1, stab_unif_estimates[stab_argsort], "b*")
    axs2.set_xticks(
        np.arange(len(stab_unif_estimates)) + 1,
        labels=trait_names[stab_argsort],
        rotation=90,
    )
    fig2.savefig(smk.output[1], format="pdf", bbox_inches="tight")
    plt.close(fig2)


if __name__ == "__main__":
    main()
