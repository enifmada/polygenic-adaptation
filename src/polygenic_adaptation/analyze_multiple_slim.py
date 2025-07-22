from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from scipy.interpolate import CubicSpline
from scipy.stats import ncx2
from tqdm import tqdm


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


def compute_avg_variance(slim_array, betas_array, stderr_array=None):
    if stderr_array is not None:
        init_variance = 2 * np.sum(slim_array[3:, 0] * (1 - slim_array[3:, 0]) * (betas_array**2 + stderr_array**2))
        final_variance = 2 * np.sum(slim_array[3:, -1] * (1 - slim_array[3:, -1]) * (betas_array**2 + stderr_array**2))
    else:
        init_variance = 2 * np.sum(slim_array[3:, 0] * (1 - slim_array[3:, 0]) * betas_array**2)
        final_variance = 2 * np.sum(slim_array[3:, -1] * (1 - slim_array[3:, -1]) * betas_array**2)
    return (init_variance + final_variance) / 2


def main():
    plt.rcParams.update(
        {
            "font.size": 11,
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": "cmr10",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": True,
        }
    )
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    coolormap = plt.get_cmap("Dark2")
    colorlist = ["#1D6996", *[coolormap(i) for i in [1, 0]], colors[3], colors[4]]
    plt.rcParams["axes.prop_cycle"] = cycler(color=colorlist)

    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", help="mode of selection")
    parser.add_argument("-dz", type=float, help="distance to optimum (directional only)")
    parser.add_argument("-h2", default=1.0, type=float, help="heritability")
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

    if smk.mode == "directional":
        # *2 b/c conversion from s2 = s to s1 = s
        BETA_EXPONENT = 1
        SCALING_FACTOR = 2
        PLOT_FACTOR = 1
    else:
        # /2 b/c the equation is S/2
        BETA_EXPONENT = 2
        SCALING_FACTOR = 0.5
        PLOT_FACTOR = -1

    betas = []
    omegas = []
    sigma_sqs = []
    str_ests = [[], []] if smk.gwas else [[]]
    loop_len = len(smk.input) - 1  # if smk.gwas else len(smk.input)

    np.random.default_rng(8)
    for input_i in tqdm(range(loop_len)):
        file_beta, file_omega, _ = get_params(smk.input[input_i])
        omegas.append(file_omega)
        slim_path = Path(smk.input[input_i]).parent.parent / "slims"
        slim_fname = Path(smk.input[input_i]).name.rpartition("_")[0] + "_slim.txt"
        slim_array = np.loadtxt(slim_path / slim_fname, skiprows=1).T

        betas_gwas_str = "_gwas" if smk.gwas else ""
        betas_path = Path(smk.input[input_i]).parent.parent / "betas"
        betas_fname = Path(smk.input[input_i]).name.rpartition("_")[0] + "_betas" + betas_gwas_str + ".txt"
        betas_alt_fname = Path(smk.input[input_i]).name.rpartition("_")[0] + "_betas.txt"
        betas_array = np.loadtxt(betas_path / betas_fname)
        if smk.gwas:
            stderrs_array = betas_array[:, 1]
            gwas_betas_array = betas_array[:, 0]
            true_betas_array = np.loadtxt(betas_path / betas_alt_fname)
            _max_betas = true_betas_array[0]
            sigma_sq = compute_avg_variance(slim_array, true_betas_array)

            # fix this later lol, h2 != 1 always
            V_E = sigma_sq * (1 - smk.h2) / smk.h2
            temp_X = (file_omega**2 + V_E) / sigma_sq
            _temp_d_over_vg = (3 + temp_X - np.sqrt(1 + 6 * temp_X + temp_X**2)) / 4

            # yeah analyze_multiple_slim and regression need to be combined into a single file, very very obviously

            # none of this makes sense with matched betas... fix
            # hist_save_parent = Path(smk.output[input_i]).parent
            # hist_save_name = Path(smk.output[input_i]).name.rpartition("_")[0]+"_betas_hist.pdf"
            # fig, axs = plt.subplots(1,1, figsize=(8,4))
            # axs.hist(betas_array, bins=20)
            # axs.axvline(true_beta, color="r", ls="--", label=r"True $\beta$")
            # axs.axvline(-true_beta, color="r", ls="--")
            # axs.axvline(true_beta*(1-temp_d_over_vg), color="g", ls="-.", label=r"Bulmer $\beta$")
            # axs.axvline(-true_beta*(1-temp_d_over_vg), color="g", ls="-.")
            # mean_plus = np.mean(betas_array[betas_array>0])
            # mean_minus = np.mean(betas_array[betas_array<=0])
            # axs.axvline(mean_plus, color="k", label="Mean")
            # axs.axvline(mean_minus, color="k")
            # axs.legend()
            # axs.set_title(rf"$\omega$ = {file_omega} $\beta$ = {true_beta} Bulmer $\beta$ = {true_beta*(1-temp_d_over_vg):.4f} mean (+) = {mean_plus:.4f} mean (-) = {mean_minus:.4f}")
            # fig.savefig(
            #     hist_save_parent/hist_save_name,
            #     format="pdf",
            #     bbox_inches="tight",
            # )
            # plt.close(fig)

            # betas_replaced = np.copy(betas_array)
            # betas_replaced[betas_replaced == 0] = true_betas_array[betas_replaced == 0]
            # fake_noisy_betas = rng.normal(true_beta*(1-temp_d_over_vg), np.mean(stderrs_array)/1.5, size=betas_array.shape[0])
            # fake_noisy_betas[fake_noisy_betas.shape[0]//2:] *= -1
            # bulmer_mean_array = np.zeros_like(true_betas_array)
            # bulmer_mean_array[:bulmer_mean_array.shape[0]//2] = mean_plus#true_beta*(1-temp_d_over_vg)
            # bulmer_mean_array[bulmer_mean_array.shape[0]//2:] = mean_minus#-true_beta*(1-temp_d_over_vg)
            betas_array = gwas_betas_array
        if smk.gwas:
            sigma_sq = compute_avg_variance(slim_array, true_betas_array)
            betas_list = [true_betas_array, betas_array]
            errors_list = [np.zeros_like(true_betas_array), stderrs_array]
            names_list = ["truebetas", "gwasbetas"]
        else:
            sigma_sq = compute_avg_variance(slim_array, betas_array)
            betas_list = [betas_array]
            errors_list = [np.zeros_like(betas_array)]
            names_list = ["betas"]

        sigma_sqs.append(sigma_sq)
        grid = np.loadtxt(smk.input[input_i])
        raw_grid = grid[0, :]
        s_unif_vals = grid[1::2, :] if smk.mode == "directional" else grid[2::2, :]
        for b_i, grid_betas in enumerate(betas_list):
            raw_s_maxes = np.argmax(s_unif_vals, axis=1)
            usable_s_locs = (raw_s_maxes > 0) & (raw_s_maxes < raw_grid.shape[0])
            if np.all(np.isclose(errors_list[b_i], 0)):
                usable_b_locs = np.ones_like(grid_betas, dtype=bool)
            else:
                usable_b_locs = errors_list[b_i] != 0
            usable_locs = usable_s_locs & usable_b_locs
            actual_betas = grid_betas[usable_locs]
            actual_berrs = errors_list[b_i][usable_locs]
            actual_s = s_unif_vals[usable_locs]
            max_signed_beta = np.max(np.abs(actual_betas))
            beta_denom = SCALING_FACTOR * max_signed_beta**BETA_EXPONENT

            expanded_raw_x = np.linspace(
                raw_grid[0],
                raw_grid[-1],
                1000,
            )
            expanded_true_x = np.linspace(raw_grid[0] / beta_denom, raw_grid[-1] / beta_denom, num=2000)

            # make this a little more general perhaps
            if smk.mode == "directional":
                good_ex_regions = np.abs(expanded_true_x) > 0.3
            else:
                bad_ex_regions = np.arange(
                    expanded_true_x.shape[0] // 2 - expanded_true_x.shape[0] // 20,
                    expanded_true_x.shape[0] // 2 + expanded_true_x.shape[0] // 20,
                )
                good_ex_regions = ~np.isin(np.arange(expanded_true_x.shape[0]), bad_ex_regions)
            # max_signed_beta = np.sqrt(np.max(betas_array**2-_stderrs_array**2))
            betas.append(max_signed_beta)
            # tf is this

            actual_lls = np.zeros((actual_s.shape[0], expanded_true_x.shape[0]))
            temp_lls = np.zeros_like(actual_lls)
            if np.all(np.isclose(actual_berrs, 0)):
                for loc in tqdm(range(actual_betas.shape[0])):
                    s_est_grid = raw_grid / (SCALING_FACTOR * actual_betas[loc] ** BETA_EXPONENT + 1e-12)
                    if actual_betas[loc] <= 0 and smk.mode == "directional":
                        ll_unif_spline = CubicSpline(s_est_grid[::-1], actual_s[loc, ::-1])
                    else:
                        ll_unif_spline = CubicSpline(s_est_grid, actual_s[loc, :])
                    ll_unif_ests = ll_unif_spline(expanded_true_x)
                    actual_lls[loc, :] = ll_unif_ests
            else:
                for loc in tqdm(range(actual_betas.shape[0])):
                    s_temp_spline = CubicSpline(raw_grid, actual_s[loc, :])
                    s_interp = s_temp_spline(expanded_raw_x)
                    if smk.mode == "directional":
                        for k in np.arange(expanded_true_x.shape[0]):
                            z_k = expanded_true_x[k]
                            temp_lls[loc, k] = np.sum(
                                s_interp
                                * (
                                    expanded_raw_x**2
                                    / (z_k**2 * np.sqrt(2 * np.pi * 4 * actual_berrs[loc] ** 2))
                                    * np.exp(
                                        -((expanded_raw_x / z_k - 2 * actual_betas[loc]) ** 2)
                                        / (2 * 4 * actual_berrs[loc] ** 2)
                                    )
                                )
                                * 1
                                / np.abs(expanded_raw_x)
                            )
                        c_spline = CubicSpline(expanded_true_x[good_ex_regions], temp_lls[loc, good_ex_regions])
                        temp_good_vals = np.copy(temp_lls[loc])
                        temp_good_vals[~good_ex_regions] = c_spline(expanded_true_x[~good_ex_regions])
                        actual_lls[loc] = temp_good_vals
                    else:
                        tc_ll = np.zeros_like(expanded_true_x)
                        for k in np.arange(expanded_true_x.shape[0]):
                            z_k = expanded_true_x[k]
                            lmbda = (actual_betas[loc] / actual_berrs[loc]) ** 2
                            temp_ncx2 = ncx2(df=1, nc=lmbda)
                            tc_ll[k] = np.sum(
                                s_interp
                                * (
                                    (expanded_raw_x**2)
                                    / ((actual_berrs[loc] ** 2) / 2 * z_k**2)
                                    * temp_ncx2.pdf(expanded_raw_x / (z_k * (actual_berrs[loc] ** 2) / 2))
                                )
                                * 1
                                / np.abs(expanded_raw_x)
                            )
                        c_spline = CubicSpline(expanded_true_x[good_ex_regions], tc_ll[good_ex_regions])
                        temp_good_vals = np.copy(tc_ll)
                        temp_good_vals[~good_ex_regions] = c_spline(expanded_true_x[~good_ex_regions])
                        actual_lls[loc, :] = temp_good_vals
            # # this no work. scales are off. loc is off. correct way of going about this is:
            # # before was LL =  PDF of RV, beta = fixed. normalize as LL/beta**2, things are fine
            # # now, we need the "pdf" of LL * 1/beta**2, where both are RVs.
            # # so figure out wtf the pdf of 1/beta**2 loks like and then convolve.
            # if np.max(errors_list[b_i]) > 0:
            #     final_lls = np.zeros_like(actual_lls)
            #     for k in tqdm(np.arange(actual_lls.shape[0])):
            #         grid_se = np.abs(4*actual_berrs[k]/np.abs(actual_betas[k]**3))
            #         norm_vari = norm(scale=grid_se)
            #         norm_pdfs = norm_vari.pdf(expanded_x[:, np.newaxis]/expanded_x)
            #         for j in np.arange(expanded_x.shape[0]):
            #             final_lls[k, j] = np.sum(actual_lls[k] * norm_pdfs[j, :] * 1/np.abs(expanded_x))
            #         final_lls[k, :] /= np.min(final_lls[k, :])
            #     actual_lls = final_lls

            summed_lls = np.sum(actual_lls, axis=0)
            unif_ll_argmax = expanded_true_x[np.argmax(summed_lls)]
            str_ests[b_i].append(unif_ll_argmax)
            fig, axs = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
            for loc in range(actual_lls.shape[0]):
                axs.plot(expanded_true_x, actual_lls[loc, :])
            axs.plot(
                expanded_true_x,
                summed_lls / actual_lls.shape[0],
                color="k",
                lw=2,
            )
            axs.set_title(f"Unif {smk.mode}")
            fig.suptitle(f"{smk.mode} max = {unif_ll_argmax:.4f}")
            temp_path = Path(smk.input[input_i])
            temp_parent = temp_path.parent.parent
            fname = "surfaces/" + temp_path.name.rpartition("_")[0] + f"_{names_list[b_i]}.pdf"
            fig.savefig(temp_parent / fname, format="pdf", bbox_inches="tight")
            plt.close(fig)

    omegas = np.array(omegas)
    betas = np.array(betas)
    sigma_sqs = np.array(sigma_sqs)
    str_ests = np.array(str_ests)

    if smk.vary == "beta":
        x_data = betas
        x_label = "Effect size"
        y_label = "True effect size"
    else:
        # x_data = smk.dz / (omegas**2)
        # x_label = r"$\alpha$"
        # y_label = r"$\hat{\alpha}$"
        x_data = omegas
        x_label = r"$\omega$"
        y_label = r"Selection gradient"

    # compute V_E from h2, Vg. for now assume V_G = V_g...? dunno.
    V_E = sigma_sqs * (1 - smk.h2) / smk.h2

    # account for bulmer - d/Vg = more complicated eq 17
    X = (omegas**2 + V_E) / sigma_sqs
    d_over_vg = (3 + X - np.sqrt(1 + 6 * X + X**2)) / 4
    if smk.mode == "directional":
        str_theory = smk.dz / (omegas**2 + V_E)
        str_theories = [str_theory] * len(str_ests)
        str_theories_labels = ["Theory"] * len(str_ests)
    elif smk.gwas:
        str_theory = 1 / (omegas**2 + sigma_sqs + V_E)
        str_theory_semibulmer = 1 / (omegas**2 + sigma_sqs * (1 - d_over_vg) + V_E)
        str_theory_bulmer = (1 - d_over_vg) ** 2 / (omegas**2 + sigma_sqs * (1 - d_over_vg) + V_E)
        str_theories = [str_theory_bulmer, str_theory_semibulmer]
        str_theories_labels = ["Theory (Bulmer)", "Theory (semi-Bulmer)"]
    else:
        str_theories = [(1 - d_over_vg) ** 2 / (omegas**2 + sigma_sqs * (1 - d_over_vg) + V_E)]
        str_theories_labels = ["Theory"]

        # weird stuff from making one plot one time
        # S_theory = str_theory/((1-d_over_vg)**2)
        # omega_sq_theory = 1/S_theory - sigma_sqs*(1-d_over_vg)
        # omega_sq = ((1-d_over_vg)**2-(1-d_over_vg)*sigma_sqs*str_theory)/str_theory
        # omega_theory = np.sqrt(omega_sq)
        # S_ests = np.abs(str_ests)/((1-d_over_vg)**2)
        # omega_sq_ests = 1/S_ests - sigma_sqs*(1-d_over_vg)
    for b_i in range(str_ests.shape[0]):
        fig, axs = plt.subplots(1, 1, figsize=(3, 3), layout="constrained")

        if smk.gwas:
            axs.plot(x_data, PLOT_FACTOR * str_ests[b_i, :], ".", label="Estimate")
            # axs.plot(x_data, str_theory_semibulmer, ".", label="Theory (semi-Bulmer)")
            axs.plot(x_data, str_theories[b_i], ".", label=str_theories_labels[b_i])

            # axs.plot(x_data, str_theory, ".", label = "Theory (NO Bulmer)")
            # axs.set_ylim([0, 50])
        else:
            if smk.mode == "directional":
                axs.plot([0, 1], [0, 1], "r--")
            axs.plot(x_data, str_ests[b_i, :], ".", label="Estimate")
            axs.plot(x_data, str_theories[b_i], ".", label=str_theories_labels[b_i])

        # for true_w in np.unique(omegas):
        #     axs.axhline(true_w, ls="--", color=colorlist[3])
        # axs.set_xlim([0, 0.5])
        # axs.set_ylim([0, 0.5])
        # axs.set_xticks([0.1, 0.2, 0.4])
        axs.set_xlabel(x_label)
        axs.set_ylabel(y_label)
        axs.legend()
        fig.savefig(smk.output[-b_i - 1], format="pdf", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
