from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.odr
from cycler import cycler
from scipy.interpolate import CubicSpline
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
    parser.add_argument("-dz", type=float, default=0.0, help="distance to optimum (directional only)")
    parser.add_argument("-h2", default=1.0, type=float, help="heritability")
    parser.add_argument("--vary", type=str, help="variable to vary")
    parser.add_argument(
        "--gwas",
        action="store_true",
        help="flag for if the betas are taken from a gwas vs ground truth",
    )
    parser.add_argument("-i", "--input", nargs="*", help="input")
    parser.add_argument("-o", "--output", nargs="*", help="output")
    parser.add_argument("--regmode", type=str, default="lsq", help="mode of regression")

    smk = parser.parse_args()

    assert smk.regmode in ("lsq", "weighted")
    assert smk.mode in ("directional", "stabilizing")
    assert smk.vary in ("beta", "omega")

    if smk.mode == "directional":
        BETA_EXPONENT = 1
        SCALING_FACTOR = -2
    else:
        BETA_EXPONENT = 2
        SCALING_FACTOR = 0.5

    betas = []
    omegas = []
    sigma_sqs = []
    str_ests = [[], []] if smk.gwas else [[]]
    loop_len = len(smk.input) - 1  # if smk.gwas else len(smk.input)

    rng = np.random.default_rng(8)
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

        # things needed to compute ground truth:
        # omegas, dz, sigma_sq
        if smk.gwas:
            stderrs_array = betas_array[:, 1]
            gwas_betas_array = betas_array[:, 0]
            true_betas_array = np.loadtxt(betas_path / betas_alt_fname)
            true_beta = true_betas_array[0]
            sigma_sq = compute_avg_variance(slim_array, true_betas_array)

            V_E = sigma_sq * (1 - smk.h2) / smk.h2
            temp_X = (file_omega**2 + V_E) / sigma_sq
            temp_d_over_vg = (3 + temp_X - np.sqrt(1 + 6 * temp_X + temp_X**2)) / 4

            if smk.mode == "directional":
                temp_theory_direc = smk.dz / (file_omega**2)
            else:
                temp_theory_semibulmer = 1 / (file_omega**2 + sigma_sq * (1 - temp_d_over_vg) + V_E)
                temp_theory_bulmer = (1 - temp_d_over_vg) ** 2 / (file_omega**2 + sigma_sq * (1 - temp_d_over_vg) + V_E)

                # none of this makes sense with matched betas... fix
            mean_plus = np.mean(gwas_betas_array[gwas_betas_array > 0])
            mean_minus = np.mean(gwas_betas_array[gwas_betas_array <= 0])
            fake_noisy_betas = rng.normal(
                true_beta * (1 - temp_d_over_vg), np.mean(stderrs_array), size=betas_array.shape[0]
            )
            fake_noisy_betas[fake_noisy_betas.shape[0] // 2 :] *= -1
            bulmer_mean_array = np.zeros_like(true_betas_array)
            bulmer_mean_array[: bulmer_mean_array.shape[0] // 2] = mean_plus  # true_beta*(1-temp_d_over_vg)
            bulmer_mean_array[bulmer_mean_array.shape[0] // 2 :] = mean_minus  # -true_beta*(1-temp_d_over_vg)

            betas_replaced = np.copy(gwas_betas_array)
            betas_replaced[betas_replaced == 0] = true_betas_array[betas_replaced == 0]
            betas_array = gwas_betas_array

        if smk.gwas:
            betas_list = [true_betas_array, betas_array]
            errors_list = [np.zeros_like(true_betas_array) + 1e-10, stderrs_array]
            names_list = ["truebetas", "gwasbetas"]
            if smk.mode == "directional":
                theory_list = [temp_theory_direc, temp_theory_direc]
            else:
                theory_list = [temp_theory_bulmer, temp_theory_semibulmer]
        else:
            sigma_sq = compute_avg_variance(slim_array, betas_array)
            betas_list = [betas_array]
            errors_list = [np.zeros_like(betas_array)]
            names_list = ["betas"]
            theory_list = [temp_theory_direc if smk.mode == "directional" else temp_theory_bulmer]

        sigma_sqs.append(sigma_sq)
        grid = np.loadtxt(smk.input[input_i])
        # _old_betas = np.zeros(grid.shape[0] - 1) + file_beta
        raw_grid = grid[0, :]
        s_unif_vals = grid[1::2, :] if smk.mode == "directional" else grid[2::2, :]

        expanded_x = np.linspace(
            raw_grid[0],
            raw_grid[-1],
            10000,
        )

        s_ests = np.zeros(s_unif_vals.shape[0])
        s_errs = np.zeros_like(s_ests)

        for loc in range(s_unif_vals.shape[0]):
            s_temp_spline = CubicSpline(raw_grid, s_unif_vals[loc, :])

            s_spline_curv = s_temp_spline.derivative(2)
            s_interp = s_temp_spline(expanded_x)
            max_loc = np.argmax(s_interp)
            s_est = expanded_x[max_loc]
            s_ests[loc] = s_est

            # variance = 1/curvature
            # hopefully this won't produce errors if we don't bounds check first?
            # revert if it does
            s_errs[loc] = 1 / np.sqrt(-s_spline_curv(s_est))
        usable_s_locs = (s_ests >= raw_grid[1]) & (s_ests <= raw_grid[-2])

        for b_i, grid_betas in enumerate(betas_list):
            if np.all(np.isclose(errors_list[b_i], 0)):
                usable_b_locs = np.ones_like(errors_list[b_i], dtype=bool)
            else:
                usable_b_locs = errors_list[b_i] != 0
            usable_locs = usable_s_locs & usable_b_locs
            # ok....
            actual_betas = grid_betas[usable_locs] ** BETA_EXPONENT
            actual_berrs = np.abs(
                BETA_EXPONENT * errors_list[b_i][usable_locs] * grid_betas[usable_locs] ** (BETA_EXPONENT - 1)
            )
            actual_s = s_ests[usable_locs]
            actual_serrs = s_errs[usable_locs]

            fig, axs = plt.subplots(1, 1, figsize=(5, 5))
            if np.all(np.isclose(actual_berrs, 0)):
                axs.scatter(actual_betas, actual_s, s=actual_serrs / np.min(actual_serrs), color="b")
            else:
                scatterplot = axs.scatter(actual_betas, actual_s, s=actual_serrs / np.min(actual_serrs), c=actual_berrs)
                fig.colorbar(scatterplot, ax=axs)
            axs.set_title(f"min - max s err: {np.min(actual_serrs):.4f} - {np.max(actual_serrs):.4f}")
            x_space = np.linspace(np.min(actual_betas) * 0.95, np.max(actual_betas) * 1.05, 500)
            axs.plot(
                x_space,
                -theory_list[b_i] * SCALING_FACTOR * x_space,
                label=rf"$s_\ell = {-theory_list[b_i]*SCALING_FACTOR:.4f}\beta_\ell{'^' + str(BETA_EXPONENT) if BETA_EXPONENT == 2 else ''}$ (theory)",
            )
            # why don't these agree?? actual regression is fucked imo
            if smk.regmode == "lsq":
                lsq_res = np.linalg.lstsq(np.vstack([actual_betas, np.zeros_like(actual_betas)]).T, actual_s)
                m_reg = lsq_res[0][0]
                # actual_regression = np.polynomial.polynomial.Polynomial.fit(grid_betas, stab_ests, deg=[2])
                # coeffs = actual_regression.convert().coef
                # axs.plot(x_space, coeffs[0] + coeffs[1] * x_space + coeffs[2] * x_space ** 2,
                #         label=rf"$s_\ell = {coeffs[0]:.4f} + {coeffs[1]:.4f}\beta_\ell + {coeffs[2]:.4f}\beta_\ell^2$")

            elif "gwas" in names_list[b_i]:
                actual_betas = grid_betas[usable_locs]
                actual_berrs = errors_list[b_i][usable_locs]

                # perform deming regression on beta^2 vs s_est;

                # oook in ODR we trust
                # have to define the function here b/c idk how to give parameters to optimization function
                def f(B, x):
                    return B[0] * x**BETA_EXPONENT

                odr_model = scipy.odr.Model(f)
                odr_data = scipy.odr.RealData(actual_betas, actual_s, sx=actual_berrs, sy=actual_serrs)
                odr_odr = scipy.odr.ODR(odr_data, odr_model, beta0=[0], maxit=1000)
                odr_output = odr_odr.run()
                # who knows lol
                m_reg = odr_output.beta[0]

                ##might be a better method of doing this but for now
                # beta_variances = actual_berrs**2
                # stab_variances = actual_serrs**2
                # lmbda = np.mean(stab_variances)/np.mean(beta_variances)
                #
                # #now compute all the necessary moments
                # syy = np.sum(actual_s**2)
                # sxx = np.sum(actual_betas**2)
                # sxy = np.sum(actual_s*actual_betas)
                #
                # slope_hat = ((syy-lmbda*sxx)+np.sqrt((lmbda*sxx-syy)**2+4*lmbda*sxy**2))/(2*sxy)
                # print(f"lambda: {lmbda:.4f} slope: {slope_hat:.4f} ODR beta: {odr_res:.4f}")
                #
                # raise Error
                # str_ests[b_i].append(2*slope_hat)
                # x_space = np.linspace(np.min(actual_betas**2) * .95, np.max(actual_betas**2) * 1.05, 500)
                # fig, axs = plt.subplots(1, 1, figsize=(5, 5))
                # axs.plot(grid_betas**2, s_ests, ".")
                # axs.plot(x_space, slope_hat*x_space, label=rf"$s_\ell = {slope_hat:.4f}\beta_\ell^2$")
                # axs.legend()
                # temp_path = Path(smk.input[input_i])
                # temp_parent = temp_path.parent.parent
                # fname = "surfaces/" + temp_path.name.rpartition("_")[
                #     0] + f"_weighted_regression_{names_list[b_i]}_wide.pdf"
                # fig.savefig(temp_parent / fname, format="pdf", bbox_inches="tight")
                # plt.close(fig)
            else:
                # for true betas we only have y errs so we just do weighted least squares
                weights = np.diag(1 / actual_serrs**2)
                A = np.vstack([actual_betas, np.zeros_like(actual_betas)]).T
                Aw = np.dot(np.sqrt(weights), A)
                Bw = np.dot(actual_s, np.sqrt(weights))
                lsq_res = np.linalg.lstsq(Aw, Bw)
                m_reg = lsq_res[0][0]
            axs.plot(
                x_space,
                m_reg * x_space,
                label=rf"$s_\ell = {m_reg:.4f}\beta_\ell{'^' + str(BETA_EXPONENT) if BETA_EXPONENT == 2 else ''}$ (empirical)",
            )
            axs.legend()
            temp_path = Path(smk.input[input_i])
            temp_parent = temp_path.parent.parent
            fname = (
                "surfaces/"
                + temp_path.name.rpartition("_")[0]
                + f"_{smk.regmode}_regression_{names_list[b_i]}_wide.pdf"
            )
            fig.savefig(temp_parent / fname, format="pdf", bbox_inches="tight")
            plt.close(fig)
            str_ests[b_i].append(m_reg / SCALING_FACTOR)

    omegas = np.array(omegas)
    betas = np.array(betas)
    sigma_sqs = np.array(sigma_sqs)
    str_ests = np.array(str_ests)

    if smk.vary == "beta":
        x_data = betas
        x_label = "Effect size"
        y_label = "True effect size"
    else:
        x_data = omegas
        x_label = r"$\omega$"
        y_label = r"Selection gradient"

    # compute V_E from h2, Vg. for now assume V_G = V_g...? dunno.
    V_E = sigma_sqs * (1 - smk.h2) / smk.h2

    # account for bulmer - d/Vg = more complicated eq 17
    X = (omegas**2 + V_E) / sigma_sqs
    d_over_vg = (3 + X - np.sqrt(1 + 6 * X + X**2)) / 4
    if smk.mode == "directional":
        str_theory = smk.dz / (omegas**2)
        str_theories = [str_theory] * len(str_ests)
        str_theories_labels = ["Theory"] * len(str_ests)
        str_combo_labels = ["True betas"]
        if len(str_ests) > 1:
            str_combo_labels.append("GWAS")
    elif smk.gwas:
        str_theory = 1 / (omegas**2 + sigma_sqs + V_E)
        str_theory_semibulmer = 1 / (omegas**2 + sigma_sqs * (1 - d_over_vg) + V_E)
        str_theory_bulmer = (1 - d_over_vg) ** 2 / (omegas**2 + sigma_sqs * (1 - d_over_vg) + V_E)
        str_theories = [str_theory_bulmer, str_theory_semibulmer]
        str_theories_labels = ["Theory (Bulmer)", "Theory (semi-Bulmer)"]
        str_combo_labels = ["True betas", "GWAS"]
    else:
        str_theories = [(1 - d_over_vg) ** 2 / (omegas**2 + sigma_sqs * (1 - d_over_vg) + V_E)]
        str_theories_labels = ["Theory"]
        str_combo_labels = ["True betas"]

    fig3, axs3 = plt.subplots(1, 1, figsize=(3, 3), layout="constrained")
    fig2, axs2 = plt.subplots(1, 1, figsize=(3, 3), layout="constrained")
    for b_i in range(str_ests.shape[0]):
        fig, axs = plt.subplots(1, 1, figsize=(3, 3), layout="constrained")
        if smk.gwas:
            axs.plot(x_data, -str_ests[b_i, :], ".", label="Estimate")
            # axs.plot(x_data, str_theory_semibulmer, ".", label="Theory (semi-Bulmer)")
            axs.plot(x_data, str_theories[b_i], ".", label=str_theories_labels[b_i])
            axs2.plot(x_data, -str_ests[b_i, :] / str_theories[b_i] - 1, ".", label=str_combo_labels[b_i])
            if smk.mode == "directional":
                axs3.plot(x_data, np.sqrt(smk.dz / (-str_ests[b_i])) / x_data - 1, ".", label=str_combo_labels[b_i])
            elif "GWAS" in str_combo_labels[b_i]:
                np.sqrt(1 / (-str_ests[b_i]) - sigma_sqs * (1 - d_over_vg) - V_E)
                axs3.plot(
                    x_data,
                    np.sqrt(1 / (-str_ests[b_i]) - sigma_sqs * (1 - d_over_vg) - V_E) / x_data - 1,
                    ".",
                    label=str_combo_labels[b_i],
                )
            else:
                axs3.plot(
                    x_data,
                    np.sqrt((1 - d_over_vg) ** 2 / (-str_ests[b_i]) - sigma_sqs * (1 - d_over_vg) - V_E) / x_data - 1,
                    ".",
                    label=str_combo_labels[b_i],
                )

            # axs.plot(x_data, str_theory, ".", label = "Theory (NO Bulmer)")
            # axs.set_ylim([0, 50])
        else:
            axs.plot(x_data, -str_ests[b_i, :], ".", label="Estimate")
            axs.plot(x_data, str_theories[b_i], ".", label=str_theories_labels[b_i])
            axs2.plot(x_data, -str_ests[b_i, :] / str_theories[b_i] - 1, ".", label=str_combo_labels[b_i])

        axs.set_xlabel(x_label)
        axs.set_ylabel(y_label)
        axs.legend()
        fig.savefig(smk.output[b_i], format="pdf", bbox_inches="tight")
        plt.close(fig)
    axs2.axhline(ls="--", lw=1.5, color="k")
    axs2.set_ylabel(r"$\frac{\hat{S}-S}{S}$")
    axs2.set_xlabel(x_label)
    axs2.legend()
    axs3.axhline(0, lw=1.5, ls="--", color="k")
    axs3.set_xlabel(x_label)
    axs3.set_ylabel(r"$\frac{\hat{\omega}-\omega}{\omega}$")
    axs3.legend()
    fig2.savefig(smk.output[-2], format="pdf", bbox_inches="tight")
    fig3.savefig(smk.output[-1], format="pdf", bbox_inches="tight")
    plt.close(fig2)
    plt.close(fig3)


if __name__ == "__main__":
    main()
