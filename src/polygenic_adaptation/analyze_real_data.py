from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.odr
from joblib import Parallel, delayed
from scipy.interpolate import CubicSpline
from tqdm import tqdm

# TODO:
# estimate omegas from S (how to deal with V_E? what is V_G? start with V_G, I guess?)
# - phenotypic variance is 1 so we can use that in the equation right??
# plot omegas, see what range we get


def compute_one_jackknife(jkb_i, locs_copy, idxs_perm, idxs_per_block, f, betas, s_ests, beta_errs, s_errs, beta_exp):
    locs_copy[idxs_perm[jkb_i * idxs_per_block : (jkb_i + 1) * idxs_per_block]] = False
    temp_odr_model = scipy.odr.Model(f, extra_args=(beta_exp,))
    temp_odr_data = scipy.odr.RealData(
        betas[locs_copy], s_ests[locs_copy], sx=beta_errs[locs_copy], sy=s_errs[locs_copy]
    )
    temp_odr_odr = scipy.odr.ODR(temp_odr_data, temp_odr_model, beta0=[1e-6], maxit=100)
    temp_odr_output = temp_odr_odr.run()
    odr_jk_ests = temp_odr_output.beta[0]

    wls_weights = np.diag(1 / s_errs[locs_copy] ** 2)

    actual_betas = (betas[locs_copy] ** beta_exp).reshape((betas[locs_copy].shape[0], 1))
    wls_jk_ests = ((actual_betas.T @ wls_weights @ s_ests[locs_copy]) / (actual_betas.T @ wls_weights @ actual_betas))[
        0
    ][0]

    return odr_jk_ests, wls_jk_ests


def main():
    parser = ArgumentParser()
    parser.add_argument("--max_jk_blocks", type=int, default=50, help="maximum number of jackknife blocks to compute")
    parser.add_argument("-i", "--input", nargs="*", help="input")
    parser.add_argument("--output_parquet", help="output parquet file to store results")
    parser.add_argument("-nc", "--num_cores", type=int, help="number of cores (to speed up jackknife)")
    parser.add_argument("--pheno_descr_file", help="file containing map between phenotype code and description")
    parser.add_argument("--trait_plots_dir", help="directory to save trait regression plots (none provided = no plots)")

    smk = parser.parse_args()

    rng = np.random.default_rng(6)

    # MIN_BETA_VALUE = 1e-5
    direc_S_estimates_odr = []
    direc_S_estimates_wls = []
    direc_S_estimates = [direc_S_estimates_odr, direc_S_estimates_wls]
    direc_S_errs_jko = []
    direc_S_errs_jkw = []
    direc_S_errs_jk = [direc_S_errs_jko, direc_S_errs_jkw]
    direc_S_errs_odr = []
    direc_S_errs_wls = []
    direc_S_errs_emp = [direc_S_errs_odr, direc_S_errs_wls]

    stab_S_estimates_odr = []
    stab_S_estimates_wls = []
    stab_S_estimates = [stab_S_estimates_odr, stab_S_estimates_wls]
    stab_S_errs_jko = []
    stab_S_errs_jkw = []
    stab_S_errs_jk = [stab_S_errs_jko, stab_S_errs_jkw]
    stab_S_errs_odr = []
    stab_S_errs_wls = []
    stab_S_errs_emp = [stab_S_errs_odr, stab_S_errs_wls]

    S_ests_list = [direc_S_estimates, stab_S_estimates]
    S_errs_list_jk = [direc_S_errs_jk, stab_S_errs_jk]
    S_errs_list_emp = [direc_S_errs_emp, stab_S_errs_emp]
    trait_names = []
    trait_num_snps = []
    if smk.pheno_descr_file:
        pheno_descr_pd = pd.read_parquet(smk.pheno_descr_file)
        tick_names = []
    num_inputs = len(smk.input)
    for grid_i in tqdm(range(num_inputs // 2)):
        undersplit = Path(smk.input[grid_i]).name.split("_")
        trait_name = undersplit[0] + "_" + undersplit[1]  # not sure this works in all cases...
        assert trait_name in Path(smk.input[grid_i + num_inputs // 2]).name
        trait_names.append(trait_name)
        if smk.pheno_descr_file:
            tick_names.append(
                pheno_descr_pd.loc[pheno_descr_pd["Phenotype Code"] == trait_name]["Phenotype Description"].to_numpy()[
                    0
                ]
            )
        grid = np.loadtxt(smk.input[grid_i])
        sumstats_file = Path(smk.input[grid_i + num_inputs // 2])
        if sumstats_file.suffix == ".csv":
            sumstats = pd.read_csv(sumstats_file)
        elif sumstats_file.suffix == ".parquet":
            sumstats = pd.read_parquet(sumstats_file)
        else:
            msg = "Invalid sumstats file type!"
            raise NotImplementedError(msg)
        trait_num_snps.append(sumstats.shape[0])
        betas = sumstats["ash_beta"].to_numpy()
        beta_errs = sumstats["ash_se"].to_numpy()
        raw_grid = grid[0, :]
        dll_unif_vals = grid[1::2, :]
        sll_unif_vals = grid[2::2, :]
        unif_vals = [dll_unif_vals, sll_unif_vals]

        beta_exponents = [1, 2]
        scaling_factors = [-2, 0.5]
        sel_names = ["directional", "stabilizing"]

        expanded_x = np.linspace(raw_grid[0], raw_grid[-1], 10000)

        for sel_type, s_vals, beta_exp, scaling_factor, S_est_list, S_err_list_jk, S_err_list_emp in zip(
            sel_names,
            unif_vals,
            beta_exponents,
            scaling_factors,
            S_ests_list,
            S_errs_list_jk,
            S_errs_list_emp,
            strict=False,
        ):
            s_ests = np.zeros(s_vals.shape[0])
            s_errs = np.zeros_like(s_ests)

            for loc in range(s_vals.shape[0]):
                s_temp_spline = CubicSpline(raw_grid, s_vals[loc, :])

                s_spline_curv = s_temp_spline.derivative(2)
                s_interp = s_temp_spline(expanded_x)
                max_loc = np.argmax(s_interp)
                s_est = expanded_x[max_loc]
                s_ests[loc] = s_est
                s_errs[loc] = 1 / np.sqrt(-s_spline_curv(s_est))
            usable_s_locs = (s_ests >= raw_grid[1]) & (s_ests <= raw_grid[-2])
            usable_b_locs = beta_errs != 0
            usable_locs = usable_s_locs & usable_b_locs
            usable_locs_idxs = np.where(usable_locs)[0]

            # I hate ODR
            def f(B, x, beta_expt):
                return B[0] * (x**beta_expt)

            odr_model = scipy.odr.Model(f, extra_args=(beta_exp,))
            beta0test = [1e-1]
            x0 = np.array([1, 2, 3])
            beta_exptest = (2,)
            test_args = (beta0test, x0)
            test_args + beta_exptest
            odr_data = scipy.odr.RealData(
                betas[usable_locs], s_ests[usable_locs], sx=beta_errs[usable_locs], sy=s_errs[usable_locs]
            )
            odr_odr = scipy.odr.ODR(odr_data, odr_model, beta0=[1e-6], maxit=200)
            odr_output = odr_odr.run()
            # who knows lol
            m_odr = odr_output.beta[0]

            odr_err = odr_output.sd_beta[0]

            wls_weights = np.diag(1 / s_errs[usable_locs] ** 2)

            actual_betas = (betas[usable_locs] ** beta_exp).reshape((betas[usable_locs].shape[0], 1))
            m_wls = (
                (actual_betas.T @ wls_weights @ s_ests[usable_locs]) / (actual_betas.T @ wls_weights @ actual_betas)
            )[0][0]

            rss = np.sum(
                (1 / s_errs[usable_locs] * (s_ests[usable_locs] - m_wls * betas[usable_locs] ** beta_exp)) ** 2
            )
            cov_inv = actual_betas.T @ wls_weights @ actual_betas
            se_wls = rss / (actual_betas.shape[0] - 1)

            wls_err = np.sqrt(se_wls / cov_inv[0, 0])

            if usable_locs_idxs.shape[0] <= smk.max_jk_blocks:
                odr_jk_ests_array = np.zeros_like(usable_locs_idxs, dtype=float)
                wls_jk_ests_array = np.zeros_like(usable_locs_idxs, dtype=float)

                for jkl_i, jk_loc in enumerate(usable_locs_idxs):
                    if jkl_i > 0:
                        assert usable_locs[usable_locs_idxs[jkl_i - 1]]
                    usable_locs[jk_loc] = False
                    temp_odr_model = scipy.odr.Model(f, extra_args=(beta_exp,))
                    temp_odr_data = scipy.odr.RealData(
                        betas[usable_locs], s_ests[usable_locs], sx=beta_errs[usable_locs], sy=s_errs[usable_locs]
                    )
                    temp_odr_odr = scipy.odr.ODR(temp_odr_data, temp_odr_model, beta0=[1e-6], maxit=100)
                    temp_odr_output = temp_odr_odr.run()
                    odr_jk_ests_array[jkl_i] = temp_odr_output.beta[0]

                    wls_weights = np.diag(1 / s_errs[usable_locs] ** 2)

                    actual_betas = (betas[usable_locs] ** beta_exp).reshape((betas[usable_locs].shape[0], 1))
                    wls_jk_ests_array[jkl_i] = (
                        (actual_betas.T @ wls_weights @ s_ests[usable_locs])
                        / (actual_betas.T @ wls_weights @ actual_betas)
                    )[0][0]

                    usable_locs[jk_loc] = True

            else:
                idxs_per_block = usable_locs_idxs.shape[0] // smk.max_jk_blocks + 1

                idxs_perm = np.copy(usable_locs_idxs)
                rng.shuffle(idxs_perm)
                if smk.num_cores > 1:
                    with Parallel(n_jobs=smk.num_cores) as parallel:
                        res = parallel(
                            delayed(compute_one_jackknife)(
                                jkb_i,
                                np.copy(usable_locs),
                                idxs_perm,
                                idxs_per_block,
                                f,
                                betas,
                                s_ests,
                                beta_errs,
                                s_errs,
                                beta_exp,
                            )
                            for jkb_i in tqdm(range(usable_locs_idxs.shape[0] // idxs_per_block + 1))
                        )
                    odr_ests = [rp[0] for rp in res]
                    jk_ests = [rp[1] for rp in res]
                    odr_jk_ests_array = np.array(odr_ests, dtype=float).flatten()
                    wls_jk_ests_array = np.array(jk_ests, dtype=float).flatten()
                else:
                    odr_jk_ests_array = np.zeros(usable_locs_idxs.shape[0] // idxs_per_block + 1, dtype=float)
                    wls_jk_ests_array = np.zeros(usable_locs_idxs.shape[0] // idxs_per_block + 1, dtype=float)
                    for jkb_i in range(odr_jk_ests_array.shape[0]):
                        usable_locs[idxs_perm[jkb_i * idxs_per_block : (jkb_i + 1) * idxs_per_block]] = False
                        temp_odr_model = scipy.odr.Model(f, extra_args=(beta_exp,))
                        temp_odr_data = scipy.odr.RealData(
                            betas[usable_locs], s_ests[usable_locs], sx=beta_errs[usable_locs], sy=s_errs[usable_locs]
                        )
                        temp_odr_odr = scipy.odr.ODR(temp_odr_data, temp_odr_model, beta0=[1e-6], maxit=100)
                        temp_odr_output = temp_odr_odr.run()
                        odr_jk_ests_array[jkb_i] = temp_odr_output.beta[0]

                        wls_weights = np.diag(1 / s_errs[usable_locs] ** 2)

                        actual_betas = (betas[usable_locs] ** beta_exp).reshape((betas[usable_locs].shape[0], 1))
                        wls_jk_ests_array[jkb_i] = (
                            (actual_betas.T @ wls_weights @ s_ests[usable_locs])
                            / (actual_betas.T @ wls_weights @ actual_betas)
                        )[0][0]
                        usable_locs[idxs_perm[jkb_i * idxs_per_block : (jkb_i + 1) * idxs_per_block]] = True
            odr_jk_err = np.sqrt((odr_jk_ests_array.shape[0] - 1) * np.var(odr_jk_ests_array))
            wls_jk_err = np.sqrt((wls_jk_ests_array.shape[0] - 1) * np.var(wls_jk_ests_array))

            actual_S_odr = -m_odr / scaling_factor
            actual_S_wls = -m_wls / scaling_factor
            actual_err_odr = abs(odr_err / scaling_factor)
            actual_err_wls = abs(wls_err / scaling_factor)
            actual_err_jko = abs(odr_jk_err / scaling_factor)
            actual_err_jkw = abs(wls_jk_err / scaling_factor)

            S_est_list[0].append(actual_S_odr)
            S_est_list[1].append(actual_S_wls)
            S_err_list_jk[0].append(actual_err_jko)
            S_err_list_jk[1].append(actual_err_jkw)
            S_err_list_emp[0].append(actual_err_odr)
            S_err_list_emp[1].append(actual_err_wls)

            if smk.trait_plots_dir:
                x_space = np.linspace(
                    0.95 * np.min(betas[usable_locs] ** beta_exp), 1.05 * np.max(betas[usable_locs] ** beta_exp), 500
                )

                fig, axs = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")
                scatterplot = axs.scatter(
                    betas[usable_locs] ** beta_exp,
                    s_ests[usable_locs],
                    s=s_errs[usable_locs] / np.min(s_errs[usable_locs]),
                    c=beta_errs[usable_locs],
                )
                fig.colorbar(scatterplot, ax=axs)
                axs.plot(
                    x_space,
                    m_odr * x_space,
                    label=rf"$s_\ell = {m_odr:.4f}\beta_\ell{'^' + str(beta_exp) if beta_exp == 2 else ''}$",
                )
                axs.plot(
                    x_space,
                    m_wls * x_space,
                    label=rf"$s_\ell = {m_wls:.4f}\beta_\ell{'^' + str(beta_exp) if beta_exp == 2 else ''}$",
                )
                axs.fill_between(
                    x_space,
                    m_odr * x_space - 1.96 * odr_err,
                    m_odr * x_space + 1.96 * odr_err,
                    alpha=0.5,
                    label="ODR err",
                )
                axs.fill_between(
                    x_space,
                    m_wls * x_space - 1.96 * wls_err,
                    m_wls * x_space + 1.96 * wls_err,
                    alpha=0.5,
                    label="WLS err",
                )
                # axs.fill_between(x_space, m_odr * x_space - 1.96 * odr_jk_err, m_odr * x_space + 1.96 * odr_jk_err, alpha = .5, label="JK err")
                axs.set_title(f"min - max s err: {np.min(s_errs[usable_locs]):.4f} - {np.max(s_errs[usable_locs]):.4f}")
                axs.legend()
                fig.savefig(
                    Path(smk.trait_plots_dir) / f"{Path(smk.input[grid_i]).stem}_{sel_type}_reg_ests.pdf",
                    format="pdf",
                    bbox_inches="tight",
                )
                plt.close(fig)

    trait_names = np.array(trait_names)
    if smk.pheno_descr_file:
        tick_names = np.array(tick_names)
    direc_S_estimates = np.array(direc_S_estimates)
    stab_S_estimates = np.array(stab_S_estimates)
    direc_S_errs_emp = np.array(direc_S_errs_emp)
    direc_S_errs_jk = np.array(direc_S_errs_jk)
    stab_S_errs_emp = np.array(stab_S_errs_emp)
    stab_S_errs_jk = np.array(stab_S_errs_jk)

    everything_df = pd.DataFrame(
        {
            "trait_name": trait_names,
            "trait_num_snps": trait_num_snps,
            "direc_odr_est": direc_S_estimates[0],
            "direc_odr_emperr": direc_S_errs_emp[0],
            "direc_odr_jkerr": direc_S_errs_jk[0],
            "direc_wls_est": direc_S_estimates[1],
            "direc_wls_emperr": direc_S_errs_emp[1],
            "direc_wls_jkerr": direc_S_errs_jk[1],
            "stab_odr_est": stab_S_estimates[0],
            "stab_odr_emperr": stab_S_errs_emp[0],
            "stab_odr_jkerr": stab_S_errs_jk[0],
            "stab_wls_est": stab_S_estimates[1],
            "stab_wls_emperr": stab_S_errs_emp[1],
            "stab_wls_jkerr": stab_S_errs_jk[1],
        }
    )
    if smk.pheno_descr_file:
        everything_df["trait_desc"] = tick_names
        cur_cols = everything_df.columns.to_list()
        new_cols = deepcopy(cur_cols)
        new_cols.insert(1, new_cols[-1])
        new_cols = new_cols[:-1]
        everything_df = everything_df[new_cols]
    everything_df.to_parquet(path=smk.output_parquet)


if __name__ == "__main__":
    main()
