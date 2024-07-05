from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", nargs="*", help="input")
    parser.add_argument("-o", "--output", nargs="*", help="output")

    smk = parser.parse_args()

    # MIN_BETA_VALUE = 1e-5
    d_unif_hits = 0
    d_unif_miss = 0
    s_unif_hits = 0
    s_unif_miss = 0
    d_EM_hits = 0
    d_EM_miss = 0
    s_EM_hits = 0
    s_EM_miss = 0
    direc_unif_estimates = []
    stab_unif_estimates = []
    direc_EM_estimates = []
    stab_EM_estimates = []
    num_inputs = len(smk.input)
    for grid_i in tqdm(range((num_inputs - 1) // 2)):
        assert (
            Path(smk.input[grid_i]).name.rpartition("_")[0]
            == Path(smk.input[grid_i + (num_inputs - 1) // 2]).name.rpartition("_")[0]
        )
        grid = np.loadtxt(smk.input[grid_i])
        betas = np.loadtxt(smk.input[grid_i + (num_inputs - 1) // 2])
        raw_grid = grid[0, :]
        dll_unif_vals = grid[1::4, :]
        sll_unif_vals = grid[2::4, :]
        dll_EM_vals = grid[3::4, :]
        sll_EM_vals = grid[4::4, :]

        max_signed_beta = np.max(np.abs(betas))
        expanded_direc_x = np.linspace(
            raw_grid[0] / (2 * max_signed_beta),
            raw_grid[-1] / (2 * max_signed_beta),
            20000,
        )
        direc_s2l_raw_1 = raw_grid[1] / (2 * max_signed_beta)
        direc_s2l_raw_2 = raw_grid[-2] / (2 * max_signed_beta)
        expanded_stab_x = np.linspace(
            raw_grid[0] / (max_signed_beta**2 / 2),
            raw_grid[-1] / (max_signed_beta**2 / 2),
            20000,
        )
        stab_s2l_raw_1 = raw_grid[1] / (max_signed_beta**2 / 2)
        stab_s2l_raw_2 = raw_grid[-2] / (max_signed_beta**2 / 2)
        summed_unif_dlls = np.zeros_like(expanded_direc_x)
        summed_unif_slls = np.zeros_like(expanded_stab_x)
        summed_EM_dlls = np.zeros_like(expanded_direc_x)
        summed_EM_slls = np.zeros_like(expanded_stab_x)
        all_dll_unif_ests = np.zeros(
            (dll_unif_vals.shape[0], summed_unif_dlls.shape[0])
        )
        all_sll_unif_ests = np.zeros(
            (dll_unif_vals.shape[0], summed_unif_dlls.shape[0])
        )
        all_dll_EM_ests = np.zeros((dll_unif_vals.shape[0], summed_unif_dlls.shape[0]))
        all_sll_EM_ests = np.zeros((dll_unif_vals.shape[0], summed_unif_dlls.shape[0]))
        for loc in range(dll_unif_vals.shape[0]):
            # *2 b/c conversion from s2 = s to s1 = s
            sdz_est_grid = raw_grid / (2 * betas[loc])
            # /2 b/c I think that's the right coefficient in the actual equations?
            s_est_grid = raw_grid / (betas[loc] ** 2 / 2)
            sll_unif_spline = PchipInterpolator(s_est_grid, sll_unif_vals[loc, :])
            sll_EM_spline = PchipInterpolator(s_est_grid, sll_EM_vals[loc, :])
            if betas[loc] >= 0:
                dll_unif_spline = PchipInterpolator(sdz_est_grid, dll_unif_vals[loc, :])
                dll_EM_spline = PchipInterpolator(sdz_est_grid, dll_EM_vals[loc, :])
            else:
                dll_unif_spline = PchipInterpolator(
                    sdz_est_grid[::-1], dll_unif_vals[loc, ::-1]
                )
                dll_EM_spline = PchipInterpolator(
                    sdz_est_grid[::-1], dll_EM_vals[loc, ::-1]
                )

            dll_unif_ests = dll_unif_spline(expanded_direc_x)
            dll_EM_ests = dll_EM_spline(expanded_direc_x)
            all_dll_unif_ests[loc, :] = dll_unif_ests
            all_dll_EM_ests[loc, :] = dll_EM_ests
            summed_unif_dlls += dll_unif_ests
            summed_EM_dlls += dll_EM_ests

            sll_unif_ests = sll_unif_spline(expanded_stab_x)
            sll_EM_ests = sll_EM_spline(expanded_stab_x)
            all_sll_unif_ests[loc, :] = sll_unif_ests
            all_sll_EM_ests[loc, :] = sll_EM_ests
            summed_unif_slls += sll_unif_ests
            summed_EM_slls += sll_EM_ests

        fig, axs = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
        for loc in range(dll_unif_vals.shape[0]):
            axs[0, 0].plot(expanded_direc_x, all_dll_unif_ests[loc, :])
            axs[0, 1].plot(expanded_direc_x, all_dll_EM_ests[loc, :])
            axs[1, 0].plot(expanded_stab_x, all_sll_unif_ests[loc, :])
            axs[1, 1].plot(expanded_stab_x, all_sll_EM_ests[loc, :])
        axs[0, 0].plot(
            expanded_direc_x,
            summed_unif_dlls / all_dll_unif_ests.shape[0],
            color="k",
            lw=2,
        )
        axs[0, 1].plot(
            expanded_direc_x, summed_EM_dlls / all_dll_EM_ests.shape[0], color="k", lw=2
        )
        axs[1, 0].plot(
            expanded_stab_x,
            summed_unif_slls / all_dll_unif_ests.shape[0],
            color="k",
            lw=2,
        )
        axs[1, 1].plot(
            expanded_stab_x, summed_EM_slls / all_dll_EM_ests.shape[0], color="k", lw=2
        )
        axs[0, 0].axvline(direc_s2l_raw_1, color="k", ls="--")
        axs[0, 0].axvline(direc_s2l_raw_2, color="k", ls="--")
        axs[0, 1].axvline(direc_s2l_raw_1, color="k", ls="--")
        axs[0, 1].axvline(direc_s2l_raw_2, color="k", ls="--")
        axs[1, 0].axvline(stab_s2l_raw_1, color="k", ls="--")
        axs[1, 0].axvline(stab_s2l_raw_2, color="k", ls="--")
        axs[1, 1].axvline(stab_s2l_raw_1, color="k", ls="--")
        axs[1, 1].axvline(stab_s2l_raw_2, color="k", ls="--")
        axs[0, 0].set_title("Unif Direc")
        axs[0, 1].set_title("EM Direc")
        axs[1, 0].set_title("Unif Stab")
        axs[1, 1].set_title("EM Stab")
        fig.suptitle(f"{grid_i}")
        fig.savefig(
            Path(smk.output[1]).parent
            / f"{Path(smk.input[grid_i]).stem}_all_pchip_ests.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(fig)
        dll_unif_max = np.max(summed_unif_dlls)
        dll_unif_argmax = expanded_direc_x[np.argmax(summed_unif_dlls)]
        dll_EM_max = np.max(summed_EM_dlls)
        dll_EM_argmax = expanded_direc_x[np.argmax(summed_EM_dlls)]

        sll_unif_max = np.max(summed_unif_slls)
        sll_unif_argmax = expanded_stab_x[np.argmax(summed_unif_slls)]
        sll_EM_max = np.max(summed_EM_slls)
        sll_EM_argmax = expanded_stab_x[np.argmax(summed_EM_slls)]

        if "dz0.0_" in smk.input[grid_i]:
            # stabilizing only
            stab_unif_estimates.append(-sll_unif_argmax)
            stab_EM_estimates.append(-sll_EM_argmax)

            if dll_unif_max > sll_unif_max:
                s_unif_miss += 1
            else:
                s_unif_hits += 1
            if dll_EM_max > sll_EM_max:
                s_EM_miss += 1
            else:
                s_EM_hits += 1

        else:
            # directional only
            direc_unif_estimates.append(dll_unif_argmax)
            direc_EM_estimates.append(dll_EM_argmax)

            if dll_unif_max > sll_unif_max:
                d_unif_hits += 1
            else:
                d_unif_miss += 1

            if dll_EM_max > sll_EM_max:
                d_EM_hits += 1
            else:
                d_EM_miss += 1

    d = {
        "Inf Direc": [f"{d_unif_hits}/{d_EM_hits}", f"{s_unif_miss}/{s_EM_miss}"],
        "Inf Stab": [f"{d_unif_miss}/{d_EM_miss}", f"{s_unif_hits}/{s_EM_hits}"],
    }

    ctable = pd.DataFrame(data=d, index=["Sim Direc", "Sim Stab"])

    ctable.to_csv(smk.output[0])

    fig, axs = plt.subplots(1, 2, figsize=(6.2, 3.1), layout="constrained")

    axs[0].boxplot(
        [direc_unif_estimates, direc_EM_estimates], labels=["direc unif", "direc EM"]
    )
    axs[1].boxplot(
        [stab_unif_estimates, stab_EM_estimates], labels=["stab unif", "stab EM"]
    )
    fig.savefig(smk.output[1], format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
