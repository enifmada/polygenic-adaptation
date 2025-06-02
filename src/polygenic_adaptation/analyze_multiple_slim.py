from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

# from cycler import cycler


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
    if beta > 0 and omega > 0:
        return beta, omega, seed
    raise ValueError


def compute_avg_variance(slim_array, betas_array):
    init_variance = 2 * np.sum(
        slim_array[3:, 0] * (1 - slim_array[3:, 0]) * betas_array**2
    )
    final_variance = 2 * np.sum(
        slim_array[3:, -1] * (1 - slim_array[3:, -1]) * betas_array**2
    )
    return (init_variance + final_variance) / 2


def main():
    # plt.rcParams.update({'font.size': 11,
    #                      'text.usetex': False,
    #                      'font.family': 'serif',
    #                      'font.serif': 'cmr10',
    #                      'mathtext.fontset': 'cm',
    #                      'axes.unicode_minus': False,
    #                      'axes.formatter.use_mathtext': True, })
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']
    #
    # coolormap = plt.get_cmap("Dark2")
    # colorlist = ["#1D6996", *[coolormap(i) for i in [1, 0]], colors[3], colors[4]]
    # init_colorlist = colorlist
    # plt.rcParams["axes.prop_cycle"] = cycler(color=colorlist)

    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", help="mode of selection")
    parser.add_argument(
        "-dz", type=float, help="distance to optimum (directional only)"
    )
    parser.add_argument("--vary", type=str, help="variable to vary")
    parser.add_argument(
        "--gwas",
        action="store_true",
        help="flag for if the betas are taken from a gwas vs ground truth",
    )
    parser.add_argument("-i", "--input", nargs="*", help="input")
    parser.add_argument("-o", "--output", nargs="*", help="output")

    smk = parser.parse_args()

    assert smk.mode in ("directional", "stabilizing")
    assert smk.vary in ("beta", "omega")

    betas = []
    omegas = []
    sigma_sqs = []
    str_ests = []
    loop_len = len(smk.input) - 1 if smk.gwas else len(smk.input)
    for input_i in range(loop_len):
        file_beta, file_omega, _ = get_params(smk.input[input_i])
        omegas.append(file_omega)
        slim_path = Path(smk.input[input_i]).parent.parent / "slims"
        slim_fname = Path(smk.input[input_i]).name.rpartition("_")[0] + "_slim.txt"
        slim_array = np.loadtxt(slim_path / slim_fname, skiprows=1).T

        betas_gwas_str = "_gwas" if smk.gwas else ""
        betas_path = Path(smk.input[input_i]).parent.parent / "betas"
        betas_fname = (
            Path(smk.input[input_i]).name.rpartition("_")[0]
            + "_betas"
            + betas_gwas_str
            + ".txt"
        )
        betas_array = np.loadtxt(betas_path / betas_fname)
        if smk.gwas:
            _stderrs_array = betas_array[:, 1]
            betas_array = betas_array[:, 0]
        sigma_sq = compute_avg_variance(slim_array, betas_array)
        sigma_sqs.append(sigma_sq)
        grid = np.loadtxt(smk.input[input_i])
        # _old_betas = np.zeros(grid.shape[0] - 1) + file_beta
        raw_grid = grid[0, :]
        dll_unif_vals = grid[1::2, :]
        sll_unif_vals = grid[2::2, :]
        grid_betas = betas_array
        max_signed_beta = np.max(np.abs(grid_betas))
        betas.append(max_signed_beta)
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
        all_dll_unif_ests = np.zeros(
            (dll_unif_vals.shape[0], summed_unif_dlls.shape[0])
        )
        all_sll_unif_ests = np.zeros(
            (dll_unif_vals.shape[0], summed_unif_dlls.shape[0])
        )
        for loc in range(dll_unif_vals.shape[0]):
            # *2 b/c conversion from s2 = s to s1 = s
            sdz_est_grid = raw_grid / (2 * grid_betas[loc])
            # no /2, it's not there lol
            s_est_grid = raw_grid / (grid_betas[loc] ** 2 / 2)
            sll_unif_spline = CubicSpline(s_est_grid, sll_unif_vals[loc, :])

            if grid_betas[loc] >= 0:
                dll_unif_spline = CubicSpline(sdz_est_grid, dll_unif_vals[loc, :])
            else:
                dll_unif_spline = CubicSpline(
                    sdz_est_grid[::-1], dll_unif_vals[loc, ::-1]
                )

            dll_unif_ests = dll_unif_spline(expanded_direc_x)
            all_dll_unif_ests[loc, :] = dll_unif_ests
            summed_unif_dlls += dll_unif_ests

            sll_unif_ests = sll_unif_spline(expanded_stab_x)
            all_sll_unif_ests[loc, :] = sll_unif_ests
            summed_unif_slls += sll_unif_ests

        np.max(summed_unif_dlls)
        dll_unif_argmax = expanded_direc_x[np.argmax(summed_unif_dlls)]

        np.max(summed_unif_slls)
        sll_unif_argmax = expanded_stab_x[np.argmax(summed_unif_slls)]

        if smk.mode == "directional":
            str_ests.append(dll_unif_argmax)
        else:
            str_ests.append(sll_unif_argmax)

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
        axs[0].plot(
            raw_grid / (2 * max_signed_beta),
            np.zeros_like(raw_grid) + np.max(all_dll_unif_ests),
            "r.",
        )
        axs[1].plot(
            expanded_stab_x,
            summed_unif_slls / all_dll_unif_ests.shape[0],
            color="k",
            lw=2,
        )
        axs[1].plot(
            raw_grid / (max_signed_beta**2 / 2),
            np.zeros_like(raw_grid) + np.max(all_sll_unif_ests),
            "r.",
        )
        axs[0].axvline(direc_s2l_raw_1, color="k", ls="--")
        axs[0].axvline(direc_s2l_raw_2, color="k", ls="--")
        axs[1].axvline(stab_s2l_raw_1, color="k", ls="--")
        axs[1].axvline(stab_s2l_raw_2, color="k", ls="--")
        axs[0].set_title("Unif Direc")
        axs[1].set_title("Unif Stab")
        fig.suptitle(
            f"Direc max = {dll_unif_argmax:.4f} Stab max = {sll_unif_argmax:.4f}"
        )
        fig.savefig(
            smk.output[input_i],
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    betas = np.array(betas)
    sigma_sqs = np.array(sigma_sqs)
    omegas = np.array(omegas)
    str_ests = np.array(str_ests)

    if smk.vary == "beta":
        x_data = betas
        x_label = "Effect size"
        y_label = "True effect size"
    else:
        x_data = omegas
        x_label = r"$\omega$"
        y_label = r"Selection gradient"

    # account for bulmer - d/Vg = more complicated eq 17
    X = (omegas**2 + 0) / sigma_sqs
    d_over_vg = (3 + X - np.sqrt(1 + 6 * X + X**2)) / 4
    if smk.mode == "directional":
        str_theory = smk.dz / (omegas**2)
    else:
        # whoops - it multiplies beta so it's squared here. Also, the V_P correction factor!
        str_theory = (1 - d_over_vg) ** 2 / (omegas**2 + sigma_sqs * (1 - d_over_vg))

        # weird stuff from making one plot one time
        # S_theory = str_theory/((1-d_over_vg)**2)
        # omega_sq_theory = 1/S_theory - sigma_sqs*(1-d_over_vg)
        # omega_sq = ((1-d_over_vg)**2-(1-d_over_vg)*sigma_sqs*str_theory)/str_theory
        # omega_theory = np.sqrt(omega_sq)
        # S_ests = np.abs(str_ests)/((1-d_over_vg)**2)
        # omega_sq_ests = 1/S_ests - sigma_sqs*(1-d_over_vg)

    fig, axs = plt.subplots(1, 1, figsize=(6, 6), layout="constrained")
    axs.plot(x_data, np.abs(str_ests), ".", label=r"Estimate")
    axs.plot(x_data, str_theory, "r.", label="Theory")
    # axs.plot([0,1],[0,1], "r--")
    # for true_w in np.unique(omegas):
    #     axs.axhline(true_w, ls="--", color=colorlist[3])
    # axs.set_xlim([0, 0.5])
    # axs.set_ylim([0, 0.5])
    # axs.set_xticks([0.1, 0.2, 0.4])
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)
    axs.legend()
    fig.savefig(smk.output[-1], format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
