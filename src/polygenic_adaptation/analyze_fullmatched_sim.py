from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.odr

# from odrpack import odr_fit
from cycler import cycler
from scipy.interpolate import CubicSpline
from tqdm import tqdm

# perhaps the variation in effect sizes needs to be accounted for? that might be why estimates are inaccurate. Should look into.
# in the Bulmer effect formula - use the averaged version in the Appendix rather than the one-size assumption
# yeah tbh this could potentially explain things? let's seeeeee


def get_params(fname, vars, prefixes):
    vars_dict = {}
    flags_dict = {}
    for var in vars:
        vars_dict[var] = 0
        flags_dict[var] = False
    fparts = fname.split("_")
    for fpart in fparts[1:]:
        if fpart[0] in prefixes and not flags_dict[vars[prefixes.index(fpart[0])]]:
            vars_dict[vars[prefixes.index(fpart[0])]] = float(fpart[1:])
            flags_dict[vars[prefixes.index(fpart[0])]] = True
        # this is awful
        elif fpart[:2] in prefixes and not flags_dict[vars[prefixes.index(fpart[:2])]]:
            vars_dict[vars[prefixes.index(fpart[:2])]] = float(fpart[2:])
            flags_dict[vars[prefixes.index(fpart[:2])]] = True
    return vars_dict


def compute_avg_variance(slim_array, betas_array, stderr_array=None):
    if stderr_array is not None:
        init_variance = 2 * np.sum(slim_array[:, 0] * (1 - slim_array[:, 0]) * (betas_array**2 + stderr_array**2))
        final_variance = 2 * np.sum(slim_array[:, -1] * (1 - slim_array[:, -1]) * (betas_array**2 + stderr_array**2))
    else:
        init_variance = 2 * np.sum(slim_array[:, 0] * (1 - slim_array[:, 0]) * betas_array**2)
        final_variance = 2 * np.sum(slim_array[:, -1] * (1 - slim_array[:, -1]) * betas_array**2)
    return (init_variance + final_variance) / 2


def main():
    plt.rcParams.update(
        {
            "font.size": 10,
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
    parser.add_argument("--beta_file", type=str, default="", help="path to beta file")
    parser.add_argument("-i", "--input", nargs="*", help="input")
    parser.add_argument("--output_parquet", help="output path")
    parser.add_argument(
        "--global_vg",
        nargs=2,
        default=["", ""],
        type=str,
        help="paths to beta and freq files if not using replicate-specific variance estimates",
    )
    parser.add_argument("--boxplot_letters", nargs=2, default=["", ""], help="subfigure labels for the boxplots")
    parser.add_argument("--sim_source", default="slim", type=str, help="slim vs polysim")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--use_omega", action="store_true", help="data is omega (rather than V_S) based")
    group.add_argument("--use_vs", action="store_true", help="data is V_S (rather than omega = sqrt(V_S)) based")

    smk = parser.parse_args()
    assert smk.mode in ("directional", "stabilizing")
    assert smk.sim_source in ("polysim", "slim")

    omega_dict_str = "omegas" if smk.use_omega else "vs"

    if smk.mode == "stabilizing":
        temp_c = "#1D6996"
        colorlist[0] = colorlist[-1]
        colorlist[-1] = temp_c
        plt.rcParams["axes.prop_cycle"] = cycler(color=colorlist)
    # assert smk.vary in ("beta", "omega")

    if smk.mode == "directional":
        BETA_EXPONENT = 1
        SCALING_FACTOR = -2
    else:
        BETA_EXPONENT = 2
        SCALING_FACTOR = 0.5

    GLOBAL_VG_FLAG = smk.global_vg[0] != ""

    if GLOBAL_VG_FLAG:
        all_betas = np.loadtxt(smk.global_vg[0])
        all_freqs = np.loadtxt(smk.global_vg[1])
        global_vg = 2 * np.sum(all_betas**2 * all_freqs * (1 - all_freqs))
        num_global_snps = all_betas.shape[0]

    inf_counter = 0

    known_vars = ["omega", "seed", "loci", "h2", "vs"]
    known_prefixes = ["w", "s", "l", "h", "vs"]
    if smk.vary not in known_vars:
        known_vars.append(smk.vary)
        for letter in smk.vary:
            if letter not in known_prefixes:
                known_prefixes.append(letter)
                break

    assert len(known_vars) == len(known_prefixes)

    betas = []
    omegas = []
    dzs = []
    h2s = []
    sigma_sqs = []
    x_vars = []
    loop_len = len(smk.input) - 1

    S_ests_true = []
    S_ests_wls = []
    S_ests_odr = []
    S_errs_true = []
    S_errs_wls = []
    S_errs_odr = []

    np.random.default_rng(8)
    for input_i in tqdm(range(loop_len)):
        vars_dict = get_params(smk.input[input_i], vars=known_vars, prefixes=known_prefixes)
        if vars_dict["omega"] > 0:
            assert smk.use_omega
            omegas.append(vars_dict["omega"])
            temp_omegasomething = vars_dict["omega"] ** 2
        else:
            assert smk.use_vs
            omegas.append(vars_dict["vs"])
            temp_omegasomething = vars_dict["vs"]
        x_vars.append(vars_dict[smk.vary])

        temp_h2 = vars_dict["h2"] if vars_dict["h2"] > 0 else smk.h2

        sim_path = Path(smk.input[input_i]).parent.parent / "sims"
        if smk.sim_source == "polysim":
            sim_fname = Path(smk.input[input_i]).name.rpartition("_grid")[0] + "_sim.npz"
            sim_array = np.load(sim_path / sim_fname)["arr_0"]
        else:
            sim_fname = Path(smk.input[input_i]).name.rpartition("_grid")[0] + "_sim.txt"
            sim_array = np.loadtxt(sim_path / sim_fname, skiprows=1).T

        betas_gwas_str = "_gwas"
        betas_path = Path(smk.input[input_i]).parent.parent / "betas"
        betas_fname = Path(smk.input[input_i]).name.rpartition("_grid")[0] + "_betas" + betas_gwas_str + ".txt"
        betas_array = np.loadtxt(betas_path / betas_fname)

        # veepees.append((sim_array[2, 0]+sim_array[2, -1])/2)

        stderrs_array = betas_array[:, 1]
        gwas_betas_array = betas_array[:, 0]
        if smk.beta_file:
            true_betas_array = np.loadtxt(smk.beta_file)
        else:
            true_betas_fname = Path(smk.input[input_i]).name.rpartition("_grid")[0] + "_betas" + ".txt"
            true_betas_array = np.loadtxt(betas_path / true_betas_fname)
        if GLOBAL_VG_FLAG:
            sigma_sq = global_vg * gwas_betas_array.shape[0] / num_global_snps
            if smk.vary == "betascale":
                sigma_sq *= x_vars[-1] ** 2
        else:
            freqs_rows_idx = 0 if smk.sim_source == "polysim" else 3
            sigma_sq = compute_avg_variance(sim_array[freqs_rows_idx:, :], true_betas_array)

        V_E = sigma_sq * (1 - temp_h2) / temp_h2
        temp_X = (temp_omegasomething + V_E) / sigma_sq
        temp_d_over_vg = (3 + temp_X - np.sqrt(1 + 6 * temp_X + temp_X**2)) / 4

        if smk.mode == "directional":
            temp_theory_semibulmer = smk.dz / temp_omegasomething
        else:
            temp_theory_semibulmer = 1 / (temp_omegasomething + sigma_sq * (1 - temp_d_over_vg) + V_E)
            (1 - temp_d_over_vg) ** 2 / (temp_omegasomething + sigma_sq * (1 - temp_d_over_vg) + V_E)

        sigma_sqs.append(sigma_sq)
        grid = np.loadtxt(smk.input[input_i])
        # _old_betas = np.zeros(grid.shape[0] - 1) + file_beta
        raw_grid = grid[0, :]
        s_unif_vals = grid[1::2, :] if smk.mode == "directional" else grid[2::2, :]

        # horrible workaround
        if np.any(~np.isfinite(s_unif_vals)):
            bad_1h_locs = np.nonzero(~np.isfinite(s_unif_vals[:, : raw_grid.shape[0] // 2]))
            if bad_1h_locs[0].shape[0] > 0:
                endbad_1h_locs = np.concatenate(
                    (np.where(np.diff(bad_1h_locs[0]) > 0)[0], [bad_1h_locs[0].shape[0] - 1])
                )
                for ebi in np.arange(endbad_1h_locs.shape[0]):
                    s_unif_vals[bad_1h_locs[0][endbad_1h_locs[ebi]], : bad_1h_locs[1][endbad_1h_locs[ebi]] + 1] = (
                        s_unif_vals[bad_1h_locs[0][endbad_1h_locs[ebi]], bad_1h_locs[1][endbad_1h_locs[ebi]] + 1]
                    )
                    inf_counter += bad_1h_locs[1][ebi]
            bad_2h_locs = np.nonzero(~np.isfinite(s_unif_vals[:, raw_grid.shape[0] // 2 :]))
            if bad_2h_locs[0].shape[0] > 0:
                endbad_2h_locs = np.concatenate(([0], np.where(np.diff(bad_2h_locs[0]) > 0)[0] + 1))
                for ebi in np.arange(endbad_2h_locs.shape[0]):
                    s_unif_vals[
                        bad_2h_locs[0][endbad_2h_locs[ebi]],
                        bad_2h_locs[1][endbad_2h_locs[ebi]] + raw_grid.shape[0] // 2 :,
                    ] = s_unif_vals[
                        bad_2h_locs[0][endbad_2h_locs[ebi]],
                        bad_2h_locs[1][endbad_2h_locs[ebi]] + raw_grid.shape[0] // 2 - 1,
                    ]
                    inf_counter += raw_grid.shape[0] - (bad_2h_locs[1][endbad_2h_locs[ebi]] + raw_grid.shape[0] // 2)
        assert np.all(np.isfinite(s_unif_vals))
        s_ests = np.zeros(s_unif_vals.shape[0])
        s_errs = np.zeros_like(s_ests)

        for loc in range(s_unif_vals.shape[0]):
            s_temp_spline = CubicSpline(raw_grid, s_unif_vals[loc, :])
            s_deriv_curve = s_temp_spline.derivative(1)
            s_spline_curv = s_temp_spline.derivative(2)
            possible_maxes = np.concatenate(
                (s_deriv_curve.roots(discontinuity=True, extrapolate=False), np.array([raw_grid[0], raw_grid[-1]]))
            )
            s_est_deriv = possible_maxes[np.argmax(s_temp_spline(possible_maxes))]
            s_ests[loc] = s_est_deriv

            # variance = 1/curvature
            # hopefully this won't produce errors if we don't bounds check first?
            # revert if it does
            s_errs[loc] = 1 / np.sqrt(-s_spline_curv(s_est_deriv))

        bottom_tenth, top_tenth = np.quantile(np.abs(gwas_betas_array), [0.1, 0.9])
        usable_s_locs = (s_ests >= raw_grid[1]) & (s_ests <= raw_grid[-2])
        usable_b_locs = (stderrs_array != 0) & (
            np.abs(gwas_betas_array) < top_tenth
        )  # & (np.abs(gwas_betas_array) > bottom_tenth)#
        usable_locs_gt = usable_s_locs
        usable_locs_gwas = usable_s_locs & usable_b_locs

        np.where(usable_locs_gwas)[0]
        np.where(usable_locs_gt)[0]

        # "ablate" the highest one - maybe it's biasing things a lot?
        # usable_locs_gwas[usable_locs_gwas_idxs[np.argmax(np.abs(gwas_betas_array[usable_locs_gwas]))]] = False
        # usable_locs_gt[usable_locs_gt_idxs[np.argmax(np.abs(true_betas_array[usable_locs_gt]))]] = False

        # handle 0s
        if np.sum(usable_locs_gt) == 0:
            S_ests_true.append(0)
            S_errs_true.append(0)
        if np.sum(usable_locs_gwas) == 0:
            S_ests_wls.append(0)
            S_errs_wls.append(0)
            S_ests_odr.append(0)
            S_errs_odr.append(0)
        h2s.append(temp_h2)
        # could be varied in the future
        dzs.append(smk.dz)

        # ok....

        actual_betas_gwas = gwas_betas_array[usable_locs_gwas] ** BETA_EXPONENT
        actual_betas_gt = true_betas_array[usable_locs_gt] ** BETA_EXPONENT
        actual_berrs_gwas = np.abs(
            BETA_EXPONENT * stderrs_array[usable_locs_gwas] * gwas_betas_array[usable_locs_gwas] ** (BETA_EXPONENT - 1)
        )
        actual_s_gwas = s_ests[usable_locs_gwas]
        actual_serrs_gwas = s_errs[usable_locs_gwas]
        actual_s_gt = s_ests[usable_locs_gt]
        actual_serrs_gt = s_errs[usable_locs_gt]

        sc_fig_gt, sc_axs_gt = plt.subplots(1, 1, figsize=(3.1, 3.1))
        sc_fig_gwas, sc_axs_gwas = plt.subplots(1, 1, figsize=(3.1, 3.1))

        sc_axs_gt.scatter(
            actual_betas_gt, actual_s_gt, color=colorlist[0], s=7 / (actual_serrs_gt / np.min(actual_serrs_gt))
        )
        # sc_axs_gt.set_title(f"min - max s err: {np.min(actual_serrs_gt):.4f} - {np.max(actual_serrs_gt):.4f}")
        gwas_scatterplot = sc_axs_gwas.scatter(
            actual_betas_gwas, actual_s_gwas, s=actual_serrs_gwas / np.min(actual_serrs_gwas), c=actual_berrs_gwas
        )
        sc_fig_gwas.colorbar(gwas_scatterplot, ax=sc_axs_gwas)
        sc_axs_gwas.set_title(f"min - max s err: {np.min(actual_serrs_gwas):.4f} - {np.max(actual_serrs_gwas):.4f}")

        x_space_gwas = np.linspace(np.min(actual_betas_gwas) * 0.95, np.max(actual_betas_gwas) * 1.05, 500)
        x_space_gt = np.linspace(np.min(actual_betas_gt) * 0.95, np.max(actual_betas_gt) * 1.05, 500)

        # oook in ODR we trust
        # have to define the function here b/c idk how to give parameters to optimization function
        if np.sum(usable_locs_gwas) > 0:

            def f(B, x, beta_expt):
                return B[0] * (x**beta_expt)

            odr_model = scipy.odr.Model(f, extra_args=(BETA_EXPONENT,))
            odr_data = scipy.odr.RealData(
                gwas_betas_array[usable_locs_gwas],
                actual_s_gwas,
                sx=stderrs_array[usable_locs_gwas],
                sy=actual_serrs_gwas,
            )
            odr_odr = scipy.odr.ODR(odr_data, odr_model, beta0=[1e-6], maxit=100)
            odr_output = odr_odr.run()
            # who knows lol
            m_odr = odr_output.beta[0]
            err_odr = odr_output.sd_beta[0]
            # revisit when odrpack accepts my pull request
            # odr_2 = odr_fit(f, actual_betas, actual_s, beta0=[1e-10], weight_x = 1./actual_berrs**2, weight_y = 1./actual_serrs**2)
            # m_reg = odr_2.beta[0]

            # wls - gwas
            wls_weights_gwas = np.diag(1 / actual_serrs_gwas**2)
            betas_wls_gwas = (actual_betas_gwas).reshape((actual_betas_gwas.shape[0], 1))
            m_wls_gwas = (
                (betas_wls_gwas.T @ wls_weights_gwas @ actual_s_gwas)
                / (betas_wls_gwas.T @ wls_weights_gwas @ betas_wls_gwas)
            )[0][0]
            rss_gwas = np.sum((1 / actual_serrs_gwas * (actual_s_gwas - m_wls_gwas * actual_betas_gwas)) ** 2)
            cov_inv_gwas = betas_wls_gwas.T @ wls_weights_gwas @ betas_wls_gwas
            se_wls_gwas = rss_gwas / (betas_wls_gwas.shape[0] - 1)
            err_wls_gwas = np.sqrt(se_wls_gwas / cov_inv_gwas[0, 0])

            S_ests_wls.append(-m_wls_gwas / SCALING_FACTOR)
            S_errs_wls.append(abs(err_wls_gwas / SCALING_FACTOR))

            S_ests_odr.append(-m_odr / SCALING_FACTOR)
            S_errs_odr.append(abs(err_odr / SCALING_FACTOR))

            sc_axs_gwas.plot(
                x_space_gwas,
                m_wls_gwas * x_space_gwas,
                label=rf"$s_\ell = {m_wls_gwas:.4f}\beta_\ell{'^' + str(BETA_EXPONENT) if BETA_EXPONENT == 2 else ''}$ (WLS)",
            )
            sc_axs_gwas.plot(
                x_space_gwas,
                m_odr * x_space_gwas,
                label=rf"$s_\ell = {m_odr:.4f}\beta_\ell{'^' + str(BETA_EXPONENT) if BETA_EXPONENT == 2 else ''}$ (ODR)",
            )
            # sc_axs_gwas.fill_between(x_space_gwas, m_wls_gwas * x_space_gwas - 1.96 * err_wls_gwas,
            # m_wls_gwas * x_space_gwas + 1.96 * err_wls_gwas, alpha=.4, label="WLS err")
            sc_axs_gwas.fill_between(
                x_space_gwas,
                m_odr * x_space_gwas - 1.96 * err_odr,
                m_odr * x_space_gwas + 1.96 * err_odr,
                alpha=0.4,
                label="ODR err",
            )
            sc_axs_gwas.plot(
                x_space_gwas,
                -temp_theory_semibulmer * SCALING_FACTOR * x_space_gwas,
                label=rf"$s_\ell = {-temp_theory_semibulmer * SCALING_FACTOR:.4f}\beta_\ell{'^' + str(BETA_EXPONENT) if BETA_EXPONENT == 2 else ''}$ (theory)",
            )
        sc_axs_gwas.set_ylim([raw_grid[0], raw_grid[-1]])
        sc_axs_gwas.legend()
        temp_path = Path(smk.input[input_i])
        temp_parent = temp_path.parent.parent
        fname = (
            "surfaces_cond/"
            + temp_path.name.rpartition("_grid")[0]
            + f"_regression_gwasbetas{'_uc' if 'uncon' in temp_path.name else ''}.pdf"
        )
        sc_fig_gwas.savefig(temp_parent / fname, format="pdf", bbox_inches="tight")
        plt.close(sc_fig_gwas)

        # for true betas we only have y errs so we just do weighted least squares
        if np.sum(usable_locs_gt) > 0:
            wls_weights_gt = np.diag(1 / actual_serrs_gt**2)
            betas_wls_gt = (actual_betas_gt).reshape((actual_betas_gt.shape[0], 1))
            m_wls_gt = (
                (betas_wls_gt.T @ wls_weights_gt @ actual_s_gt) / (betas_wls_gt.T @ wls_weights_gt @ betas_wls_gt)
            )[0][0]
            rss_gt = np.sum((1 / actual_serrs_gt * (actual_s_gt - m_wls_gt * actual_betas_gt)) ** 2)
            cov_inv_gt = betas_wls_gt.T @ wls_weights_gt @ betas_wls_gt
            se_wls_gt = rss_gt / (betas_wls_gt.shape[0] - 1)
            err_wls_gt = np.sqrt(se_wls_gt / cov_inv_gt[0, 0])

            sc_axs_gt.plot(
                x_space_gt,
                m_wls_gt * x_space_gt,
                label=rf"$s_\ell = {m_wls_gt:.2f}\beta_\ell{'^' + str(BETA_EXPONENT) if BETA_EXPONENT == 2 else ''}$",
                color=colorlist[1],
            )
            # sc_axs_gt.fill_between(x_space_gt, m_wls_gt * x_space_gt - 1.96 * err_wls_gt, m_wls_gt * x_space_gt + 1.96 * err_wls_gt, alpha=.4,label="WLS err")
            # sc_axs_gt.plot(x_space_gt, -temp_theory_bulmer * SCALING_FACTOR * x_space_gt,
            # label=rf"$s_\ell = {-temp_theory_bulmer*SCALING_FACTOR:.4f}\beta_\ell{'^' + str(BETA_EXPONENT) if BETA_EXPONENT == 2 else ''}$ (theory)")
        else:
            msg = "No usable loci for ground truth - something went seriously wrong with running the grids!"
            raise ValueError(msg)
        # sc_axs_gt.set_ylim([raw_grid[0], raw_grid[-1]])
        sc_axs_gt.set_xlabel(r"$\beta_\ell^2$")
        sc_axs_gt.set_ylabel(r"$\hat{s}_\ell$")
        sc_axs_gt.legend()
        temp_path = Path(smk.input[input_i])
        temp_parent = temp_path.parent.parent
        fname = (
            "surfaces_cond/"
            + temp_path.name.rpartition("_grid")[0]
            + f"_regression_truebetas{'_uc' if 'uncon' in temp_path.name else ''}.pdf"
        )
        sc_fig_gt.savefig(temp_parent / fname, format="pdf", bbox_inches="tight")
        plt.close(sc_fig_gt)
        S_ests_true.append(-m_wls_gt / SCALING_FACTOR)
        S_errs_true.append(abs(err_wls_gt / SCALING_FACTOR))

    labels = [
        "files",
        "modes",
        "dzs",
        "h2s",
        omega_dict_str,
        "sigma_sqs",
        "x_vars",
        "S_ests_true",
        "S_errs_true",
        "S_ests_wls",
        "S_errs_wls",
        "S_ests_odr",
        "S_errs_odr",
    ]
    omegas = np.array(omegas)
    betas = np.array(betas)
    sigma_sqs = np.array(sigma_sqs)
    # veepees = np.array(veepees)
    x_vars = np.array(x_vars)
    dzs = np.array(dzs)
    h2s = np.array(h2s)

    S_ests_true = np.array(S_ests_true)
    S_ests_wls = np.array(S_ests_wls)
    S_ests_odr = np.array(S_ests_odr)
    S_errs_true = np.array(S_errs_true)
    S_errs_wls = np.array(S_errs_wls)
    S_errs_odr = np.array(S_errs_odr)

    modes = [smk.mode] * x_vars.shape[0]
    modes = np.array(modes)

    data = [
        smk.input[:-1],
        modes,
        dzs,
        h2s,
        omegas,
        sigma_sqs,
        x_vars,
        S_ests_true,
        S_errs_true,
        S_ests_wls,
        S_errs_wls,
        S_ests_odr,
        S_errs_odr,
    ]

    data_dict = {}
    for label, datum in zip(labels, data, strict=False):
        data_dict[label] = datum

    dframe = pd.DataFrame(data_dict)
    dframe.to_parquet(path=smk.output_parquet)


if __name__ == "__main__":
    main()
