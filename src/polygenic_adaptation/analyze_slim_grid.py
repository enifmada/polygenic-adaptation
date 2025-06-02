from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

err_vals = [0.1, 0.25, 0.5, 1]

rng = np.random.default_rng(5)

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", nargs="*", help="input")
    parser.add_argument("-o", "--output", nargs="*", help="output")

    smk = parser.parse_args()
    grid = np.loadtxt(smk.input[0])
    true_betas = np.ones(grid.shape[0]-1)
    raw_grid = grid[0, :]
    dll_unif_vals = grid[1::2, :]
    sll_unif_vals = grid[2::2, :]

    for err_val in err_vals:
        betas = rng.normal(true_betas,err_val)

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

        expanded_stab_x_errcorr = np.linspace(
            raw_grid[0] / ((max_signed_beta**2 -err_val**2)/ 2),
            raw_grid[-1] / ((max_signed_beta**2 -err_val**2) / 2),
            20000,
        )
        stab_s2l_raw_1 = raw_grid[1] / (max_signed_beta**2 / 2)
        stab_s2l_raw_2 = raw_grid[-2] / (max_signed_beta**2 / 2)
        stab_s2l_raw_1_errcorr = raw_grid[1] / ((max_signed_beta**2 -err_val**2) / 2)
        stab_s2l_raw_2_errcorr = raw_grid[-2] / ((max_signed_beta**2 -err_val**2) / 2)

        summed_unif_dlls = np.zeros_like(expanded_direc_x)
        summed_unif_slls = np.zeros_like(expanded_stab_x)
        summed_unif_slls_errcorr = np.zeros_like(expanded_stab_x)
        all_dll_unif_ests = np.zeros((dll_unif_vals.shape[0], summed_unif_dlls.shape[0]))
        all_sll_unif_ests = np.zeros((dll_unif_vals.shape[0], summed_unif_dlls.shape[0]))
        all_sll_unif_ests_errcorr = np.zeros((dll_unif_vals.shape[0], summed_unif_dlls.shape[0]))
        for loc in range(dll_unif_vals.shape[0]):
            # *2 b/c conversion from s2 = s to s1 = s
            sdz_est_grid = raw_grid / (2 * betas[loc])
            # /2 b/c I think that's the right coefficient in the actual equations?
            s_est_grid = raw_grid / (betas[loc] ** 2 / 2)
            sll_unif_spline = CubicSpline(s_est_grid, sll_unif_vals[loc, :])

            s_est_grid_errcorr = raw_grid / (max((betas[loc] ** 2 -err_val**2),0.0001) / 2)
            sll_unif_spline_errcorr = CubicSpline(s_est_grid_errcorr, sll_unif_vals[loc, :])

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

            sll_unif_ests_errcorr = sll_unif_spline_errcorr(expanded_stab_x_errcorr)
            all_sll_unif_ests_errcorr[loc, :] = sll_unif_ests_errcorr
            summed_unif_slls_errcorr += sll_unif_ests_errcorr



        np.max(summed_unif_dlls)
        dll_unif_argmax = expanded_direc_x[np.argmax(summed_unif_dlls)]

        np.max(summed_unif_slls)
        sll_unif_argmax = expanded_stab_x[np.argmax(summed_unif_slls)]

        np.max(summed_unif_slls_errcorr)
        sll_unif_argmax_errcorr = expanded_stab_x_errcorr[np.argmax(summed_unif_slls_errcorr)]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
        for loc in range(dll_unif_vals.shape[0]):
            axs[0].plot(expanded_direc_x, all_dll_unif_ests[loc, :])
            axs[1].plot(expanded_stab_x, all_sll_unif_ests[loc, :])
            axs[2].plot(expanded_stab_x_errcorr, all_sll_unif_ests_errcorr[loc, :])
        axs[0].plot(expanded_direc_x, summed_unif_dlls / all_dll_unif_ests.shape[0], color="k", lw=2,)
        axs[1].plot(expanded_stab_x, summed_unif_slls / all_dll_unif_ests.shape[0], color="k", lw=2,)
        axs[2].plot(expanded_stab_x_errcorr, summed_unif_slls_errcorr / all_dll_unif_ests.shape[0], color="k", lw=2,)
        axs[0].axvline(direc_s2l_raw_1, color="k", ls="--")
        axs[0].axvline(direc_s2l_raw_2, color="k", ls="--")
        axs[1].axvline(stab_s2l_raw_1, color="k", ls="--")
        axs[1].axvline(stab_s2l_raw_2, color="k", ls="--")
        axs[2].axvline(stab_s2l_raw_1_errcorr, color="k", ls="--")
        axs[2].axvline(stab_s2l_raw_2_errcorr, color="k", ls="--")
        axs[0].set_title("Unif Direc")
        axs[1].set_title("Unif Stab")
        axs[1].set_title("Unif Stab Errcorr")
        fig.suptitle(f"Direc max = {dll_unif_argmax:.4f} Stab max = {sll_unif_argmax:.4f} Errcorr stab max = {sll_unif_argmax_errcorr:.4f}")
        fig.savefig(Path(smk.output[0]).with_stem(Path(smk.output[0]).name+f"{err_val}err.pdf"), format="pdf", bbox_inches="tight",)
        plt.close(fig)


if __name__ == "__main__":
    main()
