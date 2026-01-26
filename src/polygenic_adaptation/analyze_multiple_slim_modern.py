from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from scipy.interpolate import CubicSpline
from scipy.stats import ncx2
from tqdm import tqdm


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
        #this is awful
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
    parser.add_argument("--output_parquet", help="output path")
    parser.add_argument("--beta_file", type=str, default="", help="path to beta file")
    parser.add_argument("--global_vg", nargs=2, default=["", ""], type=str, help="paths to beta and freq files if not using replicate-specific variance estimates")
    parser.add_argument("--sim_source", default="slim", type=str, help="slim vs polysim")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--use_omega", action="store_true", help="data is omega (rather than V_S) based")
    group.add_argument("--use_vs", action="store_true", help="data is V_S (rather than omega = sqrt(V_S)) based")

    smk = parser.parse_args()

    assert smk.mode in ("directional", "stabilizing")
    #assert smk.vary in ("beta", "omega")

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


    omega_dict_str = "omegas" if smk.use_omega else "vs"

    known_vars = ["omega", "seed", "loci", "h2", "vs"]
    known_prefixes = ["w", "s", "l", "h", "vs"]

    GLOBAL_VG_FLAG = smk.global_vg[0] != ""

    if GLOBAL_VG_FLAG:
        all_betas = np.loadtxt(smk.global_vg[0])
        all_freqs = np.loadtxt(smk.global_vg[1])
        global_vg = 2 * np.sum(all_betas ** 2 * all_freqs * (1 - all_freqs))
        num_global_snps = all_betas.shape[0]

    betas = []
    omegas = []
    dzs = []
    h2s = []
    sigma_sqs = []
    x_vars = []
    str_ests = [[], []] if smk.gwas else [[]]
    loop_len = len(smk.input) - 1  # if smk.gwas else len(smk.input)
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

        betas_gwas_str = "_gwas" if smk.gwas else ""
        betas_path = Path(smk.input[input_i]).parent.parent / "betas"
        betas_fname = Path(smk.input[input_i]).name.rpartition("_grid")[0] + "_betas" + betas_gwas_str + ".txt"
        betas_alt_fname = Path(smk.input[input_i]).name.rpartition("_grid")[0] + "_betas.txt"
        betas_array = np.loadtxt(betas_path / betas_fname)
        if smk.gwas:
            stderrs_array = betas_array[:, 1]
            gwas_betas_array = betas_array[:, 0]
            true_betas_array = np.loadtxt(betas_path / betas_alt_fname)
            _max_betas = true_betas_array[0]
            if GLOBAL_VG_FLAG:
                sigma_sq = global_vg * gwas_betas_array.shape[0] / num_global_snps
            else:
                freqs_rows_idx = 0 if smk.sim_source == "polysim" else 3
                sigma_sq = compute_avg_variance(sim_array[freqs_rows_idx:, :], true_betas_array)
            
            V_E = sigma_sq * (1 - smk.h2) / smk.h2
            temp_X = (temp_omegasomething + V_E) / sigma_sq
            _temp_d_over_vg = (3 + temp_X - np.sqrt(1 + 6 * temp_X + temp_X**2)) / 4
            betas_array = gwas_betas_array
            betas_list = [true_betas_array, betas_array]
            errors_list = [np.zeros_like(true_betas_array), stderrs_array]
            names_list = ["truebetas", "gwasbetas"]
        else:
            sigma_sq = compute_avg_variance(sim_array, betas_array)
            betas_list = [betas_array]
            errors_list = [np.zeros_like(betas_array)]
            names_list = ["betas"]

        h2s.append(temp_h2)
        # could be varied in the future
        dzs.append(smk.dz)

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
            expanded_true_x = np.linspace(raw_grid[0] / beta_denom, raw_grid[-1] / beta_denom, num=1000)

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
                for loc in range(actual_betas.shape[0]):
                    s_est_grid = raw_grid / (SCALING_FACTOR * actual_betas[loc] ** BETA_EXPONENT + 1e-12)
                    if actual_betas[loc] <= 0 and smk.mode == "directional":
                        ll_unif_spline = CubicSpline(s_est_grid[::-1], actual_s[loc, ::-1])
                    else:
                        ll_unif_spline = CubicSpline(s_est_grid, actual_s[loc, :])
                    ll_unif_ests = ll_unif_spline(expanded_true_x)
                    actual_lls[loc, :] = ll_unif_ests
            else:
                for loc in range(actual_betas.shape[0]):
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
            comp_spline = CubicSpline(expanded_true_x, summed_lls)
            possible_maxes = np.concatenate((comp_spline.derivative(1).roots(discontinuity=True, extrapolate=False), np.array([expanded_true_x[0], expanded_true_x[-1]])))
            unif_ll_maxloc = possible_maxes[np.argmax(comp_spline(possible_maxes))]
            str_ests[b_i].append(unif_ll_maxloc)
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
            fig.suptitle(f"{smk.mode} max = {unif_ll_maxloc:.4f}")
            temp_path = Path(smk.input[input_i])
            temp_parent = temp_path.parent.parent
            fname = "surfaces_cond/" + temp_path.name.rpartition("_")[0] + f"_{names_list[b_i]}.pdf"
            fig.savefig(temp_parent / fname, format="pdf", bbox_inches="tight")
            plt.close(fig)

    omegas = np.array(omegas)
    betas = np.array(betas)
    sigma_sqs = np.array(sigma_sqs)
    str_ests = np.array(str_ests)

    str_ests *= PLOT_FACTOR

    labels = ["files", "modes", "dzs", "h2s", omega_dict_str, "sigma_sqs", "x_vars", "S_ests_true", "S_errs_true",
              "S_ests_wls", "S_errs_wls", "S_ests_odr", "S_errs_odr"]
    omegas = np.array(omegas)
    betas = np.array(betas)
    sigma_sqs = np.array(sigma_sqs)
    # veepees = np.array(veepees)
    x_vars = np.array(x_vars)
    dzs = np.array(dzs)
    h2s = np.array(h2s)

    S_ests_true = np.array(str_ests[0])
    S_ests_wls = np.array(str_ests[1])
    S_ests_odr = S_ests_wls
    S_errs_true = np.ones_like(S_ests_true)
    S_errs_wls = np.ones_like(S_ests_wls)
    S_errs_odr = np.ones_like(S_ests_odr)

    modes = [smk.mode] * x_vars.shape[0]
    modes = np.array(modes)

    datas = [smk.input[:-1], modes, dzs, h2s, omegas, sigma_sqs, x_vars, S_ests_true, S_errs_true, S_ests_wls,
             S_errs_wls, S_ests_odr, S_errs_odr]

    data_dict = {}
    for label, data in zip(labels, datas, strict=False):
        data_dict[label] = data

    dframe = pd.DataFrame(data_dict)
    dframe.to_parquet(path=smk.output_parquet)


if __name__ == "__main__":
    main()
