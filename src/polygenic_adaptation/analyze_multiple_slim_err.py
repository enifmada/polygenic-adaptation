from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator  # CubicSpline


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


def compute_avg_variance(slim_array, betas_array):
    init_variance = 2 * np.sum(slim_array[3:, 0] * (1 - slim_array[3:, 0]) * betas_array**2)
    final_variance = 2 * np.sum(slim_array[3:, -1] * (1 - slim_array[3:, -1]) * betas_array**2)
    return (init_variance + final_variance) / 2


def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", help="mode of selection")
    parser.add_argument("-dz", type=float, help="distance to optimum (directional only)")
    parser.add_argument("--vary", type=str, help="variable to vary")
    parser.add_argument("-i", "--input", nargs="*", help="input")
    parser.add_argument("-o", "--output", nargs="*", help="output")

    smk = parser.parse_args()

    assert smk.mode in ("directional", "stabilizing")
    assert smk.vary in ("beta", "omega")
    rng = np.random.default_rng(6)

    correction_types = ["neut", "plus", "minus"]
    correction_types = ["real"]

    for frac_err in [0.0, 0.1, 0.2, 0.5]:
        betas = []
        omegas = []
        sigma_sqs = []
        str_ests = {}
        for ct in correction_types:
            str_ests[ct] = []
        for input_i in range(len(smk.input)):
            _, file_omega, file_seed = get_params(smk.input[input_i])
            omegas.append(file_omega)
            slim_path = Path(smk.input[input_i]).parent.parent / "slims"
            slim_fname = Path(smk.input[input_i]).name.rpartition("_")[0] + "_slim.txt"
            slim_array = np.loadtxt(slim_path / slim_fname, skiprows=1).T

            betas_path = Path(smk.input[input_i]).parent.parent / "betas"
            betas_fname = Path(smk.input[input_i]).name.rpartition("_")[0] + "_betas.txt"
            betas_array = np.loadtxt(betas_path / betas_fname)

            betas.append(np.max(np.abs(betas_array)))

            sigma_sq = compute_avg_variance(slim_array, betas_array)
            sigma_sqs.append(sigma_sq)

            cur_X = (file_omega**2 + 0) / sigma_sq
            cur_bulmer = 1 - (3 + cur_X - np.sqrt(1 + 6 * cur_X + cur_X**2)) / 4

            grid = np.loadtxt(smk.input[input_i])
            raw_grid = grid[0, :]
            dll_unif_vals = grid[1::2, :]
            sll_unif_vals = grid[2::2, :]
            grid_betas = np.zeros_like(betas_array)
            for i in np.arange(betas_array.shape[0]):
                grid_betas[i] = rng.normal(betas_array[i], np.abs(frac_err * betas_array[i]))
            max_signed_beta = np.max(np.abs(grid_betas))
            # [cur_bulmer ** 2, cur_bulmer ** 2 * (1 + frac_err ** 2), cur_bulmer ** 2 * (1 - frac_err ** 2)]
            for corr_factor, gb_text in zip(
                [(1 + frac_err**2) * (cur_bulmer**2) - 2 * frac_err**2],
                correction_types,
                strict=False,
            ):
                expanded_direc_x = np.linspace(
                    raw_grid[0] / (2 * max_signed_beta),
                    raw_grid[-1] / (2 * max_signed_beta),
                    10000,
                )
                direc_s2l_raw_1 = raw_grid[1] / (2 * max_signed_beta)
                direc_s2l_raw_2 = raw_grid[-2] / (2 * max_signed_beta)

                expanded_stab_x = np.linspace(
                    raw_grid[0] / (max_signed_beta**2 * corr_factor),
                    raw_grid[-1] / (max_signed_beta**2 * corr_factor),
                    10000,
                )
                stab_s2l_raw_1 = raw_grid[1] / (max_signed_beta**2 * corr_factor)
                stab_s2l_raw_2 = raw_grid[-2] / (max_signed_beta**2 * corr_factor)

                summed_unif_dlls = np.zeros_like(expanded_direc_x)
                summed_unif_slls = np.zeros_like(expanded_stab_x)
                all_dll_unif_ests = np.zeros((dll_unif_vals.shape[0], summed_unif_dlls.shape[0]))
                all_sll_unif_ests = np.zeros((dll_unif_vals.shape[0], summed_unif_dlls.shape[0]))
                for loc in range(dll_unif_vals.shape[0]):
                    # *2 b/c conversion from s2 = s to s1 = s
                    sdz_est_grid = raw_grid / (2 * grid_betas[loc])
                    # /2 b/c I think that's the right coefficient in the actual equations?
                    s_est_grid = raw_grid / (grid_betas[loc] ** 2 * corr_factor)
                    sll_unif_spline = PchipInterpolator(s_est_grid, sll_unif_vals[loc, :])

                    if grid_betas[loc] >= 0:
                        dll_unif_spline = PchipInterpolator(sdz_est_grid, dll_unif_vals[loc, :])
                    else:
                        dll_unif_spline = PchipInterpolator(sdz_est_grid[::-1], dll_unif_vals[loc, ::-1])

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
                    str_ests[gb_text].append(dll_unif_argmax)
                else:
                    str_ests[gb_text].append(sll_unif_argmax)

                if frac_err <= 0:
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
                        raw_grid / (max_signed_beta**2 * corr_factor / 2),
                        np.zeros_like(raw_grid) + np.max(all_sll_unif_ests),
                        "r.",
                    )
                    axs[0].axvline(direc_s2l_raw_1, color="k", ls="--")
                    axs[0].axvline(direc_s2l_raw_2, color="k", ls="--")
                    axs[1].axvline(stab_s2l_raw_1, color="k", ls="--")
                    axs[1].axvline(stab_s2l_raw_2, color="k", ls="--")
                    axs[0].set_title("Unif Direc")
                    axs[1].set_title("Unif Stab")
                    fig.suptitle(f"Direc max = {dll_unif_argmax:.4f} Stab max = {sll_unif_argmax:.4f}")
                    output_path = Path(smk.output[input_i])
                    err_str = f"err{frac_err}_{gb_text}" if frac_err > 0 else ""
                    fig.savefig(
                        output_path.with_stem(output_path.stem + err_str),
                        format="pdf",
                        bbox_inches="tight",
                    )
                    plt.close(fig)
        betas = np.array(betas)
        sigma_sqs = np.array(sigma_sqs)
        omegas = np.array(omegas)

        if smk.vary == "beta":
            x_data = betas
            x_label = "Effect size"
        else:
            x_data = omegas
            x_label = r"$\omega$"

        # account for bulmer - d/Vg = more complicated eq 17
        # X = (omegas**2 + 0) / sigma_sqs
        # 1 - (3 + X - np.sqrt(1 + 6 * X + X**2)) / 4

        # already accounted for the Bulmer effect, hopefully?
        str_theory = smk.dz / (omegas**2) if smk.mode == "directional" else 1 / (omegas**2 + sigma_sqs)

        fig, axs = plt.subplots(1, 1, figsize=(6, 6), layout="constrained")
        for ct in correction_types:
            axs.plot(x_data, np.abs(np.array(str_ests[ct])), ".", label=f"Estimate ({ct})")
        axs.plot(x_data, str_theory, "r.", label="Theory")
        axs.set_xlabel(x_label)
        axs.legend()
        output_path = Path(smk.output[-1])
        err_str = f"err{frac_err}" if frac_err > 0 else ""
        fig.savefig(
            output_path.with_stem(output_path.stem + err_str),
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
