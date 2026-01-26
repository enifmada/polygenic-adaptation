from __future__ import annotations

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from scipy.stats import norm

# TODO:
# estimate omegas from S (how to deal with V_E? what is V_G? start with V_G, I guess?)
# - phenotypic variance is 1 so we can use that in the equation right??
# plot omegas, see what range we get


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
    parser.add_argument("-o", "--output", nargs="*", help="output")
    parser.add_argument("--parquet_file", help="file created by analyze_real_data.py")
    parser.add_argument(
        "--no_below_zero",
        action="store_true",
        help="use this flag to set values below 0 to 0 because they are not real",
    )

    smk = parser.parse_args()

    everything_df = pd.read_parquet(smk.parquet_file)

    base_p_val = 0.05
    corr_p_val = base_p_val / (everything_df.shape[0])
    plt_i = 0
    ylims = [0, 0]

    for s_i, S_str in enumerate(["direc", "stab"]):
        S_est_odr = everything_df[f"{S_str}_odr_est"].to_numpy()
        S_est_wls = everything_df[f"{S_str}_wls_est"].to_numpy()
        S_err_odr = everything_df[f"{S_str}_odr_emperr"].to_numpy()
        S_err_wls = everything_df[f"{S_str}_wls_emperr"].to_numpy()
        S_err_jko = everything_df[f"{S_str}_odr_jkerr"].to_numpy()
        S_err_jkw = everything_df[f"{S_str}_wls_jkerr"].to_numpy()

        S_argsort = np.argsort(S_est_odr)
        z_argsort = np.argsort(S_est_odr / S_err_odr)
        np.argsort(everything_df["stab_odr_est"].to_numpy())
        # stab_S_estimates = stab_S_estimates[np.abs(stab_S_estimates)<500]

        if smk.no_below_zero and S_str == "stab":
            S_err_odr[S_est_odr < 0] = 1e-10
            S_err_jko[S_est_odr < 0] = 1e-10
            S_err_wls[S_est_wls < 0] = 1e-10
            S_err_jkw[S_est_wls < 0] = 1e-10
            S_est_odr[S_est_odr < 0] = 0
            S_est_wls[S_est_wls < 0] = 0

        max_odr = np.max(S_est_odr + np.maximum(S_err_odr, S_err_jko))
        max_wls = np.max(S_est_wls + np.maximum(S_err_wls, S_err_jkw))
        min_odr = np.min(S_est_odr - np.minimum(S_err_odr, S_err_jko))
        min_wls = np.min(S_est_wls - np.minimum(S_err_wls, S_err_jkw))
        yext = max(max_odr, max_wls) - min(min_odr, min_wls)
        if ylims[s_i]:
            yext = min(yext, 2 * ylims[s_i])
        ps_odr = 2 * norm.cdf(-np.abs(S_est_odr[S_argsort] / S_err_odr[S_argsort]))
        ps_wls = 2 * norm.cdf(-np.abs(S_est_wls[S_argsort] / S_err_wls[S_argsort]))
        ps_jko = 2 * norm.cdf(-np.abs(S_est_odr[S_argsort] / S_err_jko[S_argsort]))
        ps_jkw = 2 * norm.cdf(-np.abs(S_est_wls[S_argsort] / S_err_jkw[S_argsort]))

        sig_ps_odr = np.where(ps_odr < corr_p_val)[0]
        sig_ps_wls = np.where(ps_wls < corr_p_val)[0]
        np.where(ps_jko < corr_p_val)[0]
        np.where(ps_jkw < corr_p_val)[0]
        fig, axs = plt.subplots(1, 1, figsize=(6.1, 6.1), layout="constrained")
        axs.plot(np.arange(everything_df.shape[0]) + 0.85, S_est_odr[S_argsort], "bo")
        axs.plot(np.arange(everything_df.shape[0]) + 1.15, S_est_wls[S_argsort], "go")
        axs.errorbar(
            np.arange(everything_df.shape[0]) + 0.8,
            S_est_odr[S_argsort],
            yerr=S_err_odr[S_argsort],
            fmt="none",
            ecolor="k",
            label="ODR",
        )
        # axs.errorbar(np.arange(everything_df.shape[0])+0.9, S_est_odr[S_argsort], yerr=S_err_jko[S_argsort], fmt="none", ecolor="r", label="JK-ODR")
        axs.errorbar(
            np.arange(everything_df.shape[0]) + 1.1,
            S_est_wls[S_argsort],
            yerr=S_err_wls[S_argsort],
            fmt="none",
            ecolor="m",
            label="WLS",
        )
        # axs.errorbar(np.arange(everything_df.shape[0]) + 1.2, S_est_wls[S_argsort], yerr=S_err_jkw[S_argsort], fmt="none", ecolor="c",label="JK-Wreg")
        axs.set_xticks(
            np.arange(everything_df.shape[0]) + 1,
            labels=everything_df["trait_desc"].to_numpy()[S_argsort]
            if "trait_desc" in everything_df.columns
            else everything_df["trait_name"].to_numpy()[S_argsort],
            rotation=90,
        )
        axs.plot(
            (np.arange(everything_df.shape[0]) + 0.8)[sig_ps_odr],
            (S_est_odr[S_argsort] + S_err_odr[S_argsort] + 0.01 * yext)[sig_ps_odr],
            "k*",
        )
        # axs.plot((np.arange(everything_df.shape[0]) + 0.9)[sig_ps_jko],
        # (S_est_odr[S_argsort] + S_err_jko[S_argsort] + 0.01 * yext)[sig_ps_jko], "r*")
        axs.plot(
            (np.arange(everything_df.shape[0]) + 1.1)[sig_ps_wls],
            (S_est_wls[S_argsort] + S_err_wls[S_argsort] + 0.01 * yext)[sig_ps_wls],
            "m*",
        )
        # axs.plot((np.arange(everything_df.shape[0]) + 1.2)[sig_ps_jkw],
        # (S_est_wls[S_argsort] + S_err_jkw[S_argsort] + 0.01 * yext)[sig_ps_jkw], "c*")
        axs.legend()
        axs.axhline(ls="--", color="k", lw=1)
        if ylims[s_i]:
            axs.set_ylim([-ylims[s_i], ylims[s_i]])
        axs.set_ylabel("Selection gradient")
        fig.savefig(smk.output[plt_i], format="pdf", bbox_inches="tight")
        plt.close(fig)

        plt_i += 1

        fig2, axs2 = plt.subplots(1, 1, figsize=(6.1, 6.1), layout="constrained")
        axs2.plot(np.arange(everything_df.shape[0]) + 0.95, (S_est_odr / S_err_odr)[z_argsort], "bo", label="ODR")
        axs2.plot(np.arange(everything_df.shape[0]) + 1.05, (S_est_wls / S_err_wls)[z_argsort], "go", label="WLS")
        axs2.axhline(norm.ppf(corr_p_val / 2), ls="--", lw=2, color="r")
        axs2.axhline(-norm.ppf(corr_p_val / 2), ls="--", lw=2, color="r")
        axs2.axhline(0, ls="--", lw=1, color="k")
        axs2.set_xticks(
            np.arange(everything_df.shape[0]) + 1,
            labels=everything_df["trait_desc"].to_numpy()[z_argsort]
            if "trait_desc" in everything_df.columns
            else everything_df["trait_name"].to_numpy()[z_argsort],
            rotation=90,
        )
        axs2.legend()
        axs2.set_ylabel("z-score")
        fig2.savefig(smk.output[plt_i], format="pdf", bbox_inches="tight")
        plt.close(fig2)

        plt_i += 1

    stab_w_estimates_odr = np.sqrt(1 / everything_df["stab_odr_est"].to_numpy() - 1)
    stab_w_argsort = np.argsort(stab_w_estimates_odr)

    stab_w_estimates_wls = np.sqrt(1 / everything_df["stab_wls_est"].to_numpy() - 1)

    fig3, axs3 = plt.subplots(
        1, 1, figsize=(everything_df.shape[0] / 5, everything_df.shape[0] / 5), layout="constrained"
    )
    axs3.plot(np.arange(everything_df.shape[0]) + 0.9, stab_w_estimates_odr[stab_w_argsort], "*", label="ODR")
    axs3.plot(np.arange(everything_df.shape[0]) + 1.1, stab_w_estimates_wls[stab_w_argsort], "*", label="W-Reg")
    axs3.set_xticks(
        np.arange(everything_df.shape[0]) + 1,
        labels=everything_df["trait_desc"].to_numpy()[stab_w_argsort]
        if "trait_desc" in everything_df.columns
        else everything_df["trait_name"].to_numpy()[stab_w_argsort],
        rotation=90,
    )
    axs3.legend()
    fig3.savefig(smk.output[-1], format="pdf", bbox_inches="tight")
    plt.close(fig3)


if __name__ == "__main__":
    main()
