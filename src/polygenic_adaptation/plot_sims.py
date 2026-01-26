from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

#from odrpack import odr_fit
from cycler import cycler
from pandas import DataFrame as pdDataFrame
from scipy.stats import norm
from seaborn import boxplot as snsboxplot

#perhaps the variation in effect sizes needs to be accounted for? that might be why estimates are inaccurate. Should look into.
#in the Bulmer effect formula - use the averaged version in the Appendix rather than the one-size assumption
#yeah tbh this could potentially explain things? let's seeeeee

#things to put into the parquet: dz, h2, mode?

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
    parser.add_argument("--vary", type=str, help="variable to vary")
    parser.add_argument("--input_parquets", nargs="*", help="input analysis file(s)")
    parser.add_argument("--labels", nargs="*", help="labels for each input file")
    parser.add_argument("--output_names", nargs="*", help="output names for each plot type - full paths will be inferred")
    parser.add_argument("--group_by_reg_type", action="store_true", help="make one plot for each regression type, with one box per label. otherwise make one set of two boxplots per label")
    parser.add_argument("--boxplot_letters", nargs=2, default=["", ""], help="subfigure labels for the boxplots")

    smk = parser.parse_args()
    assert len(smk.labels) == len(smk.input_parquets)
    assert len(smk.output_names) == (5 if smk.group_by_reg_type else 6)


    #assert smk.vary in ("beta", "omega")

    dframes = []
    for d_i in range(len(smk.input_parquets)):
        dframe_i = pd.read_parquet(smk.input_parquets[d_i])
        dframe_i["labels"] = smk.labels[d_i]
        if "rescaled" in smk.labels[d_i]:
            for col in dframe_i.columns:
                if "S_ests" in col:
                    dframe_i[col] *= (float(smk.labels[d_i].rpartition("_")[-1]))**2
                elif "sigma_sq" in col:
                    dframe_i[col] /= (float(smk.labels[d_i].rpartition("_")[-1])) ** 2
                elif "omega" in col or "x_var" in col:
                    dframe_i[col] /= float(smk.labels[d_i].rpartition("_")[-1])
        dframes.append(dframe_i)


    dframe = pd.concat(dframes, axis=0, ignore_index=True)
    assert dframe["modes"].eq(dframe["modes"].iloc[0]).all(axis=0)
    mode = dframe.loc[0, "modes"]
    #check that all values in mode column are the same and equal to one value
    #set mode to that value
    LL_FLAG = False
    if (np.all(dframe["S_ests_wls"] == dframe["S_ests_odr"])):
        if (np.all(dframe["S_errs_wls"] == 1)) and (np.all(dframe["S_errs_odr"] == 1)):
            LL_FLAG = True
        else:
            msg = "Invalid input - all WLS and ODR values are identical!"
            raise ValueError(msg)

    if mode == "stabilizing":
        temp_c = "#1D6996"
        colorlist[0] = colorlist[-1]
        colorlist[-1] = temp_c
        plt.rcParams["axes.prop_cycle"] = cycler(color=colorlist)

    if smk.vary == "beta":
        x_label = "Effect size"
        y_label = "True effect size"
    elif smk.vary == "omega":
        x_label = r"$\omega$"
        y_label = r"Selection gradient"
    elif smk.vary == "vs":
        x_label = r"$V_S$"
        y_label = r"Selection gradient"
    else:
        x_label = smk.vary
        y_label = "Selection gradient"

    # compute V_E from h2, Vg. for now assume V_G = V_g...? dunno.
    V_E = dframe["sigma_sqs"] * (1 - dframe["x_vars"]) / dframe["x_vars"] if smk.vary == "h2" else dframe["sigma_sqs"] * (1 - dframe["h2s"]) / dframe["h2s"]

    omegasomething = dframe["omegas"] ** 2 if "omegas" in dframe else dframe["vs"]
    # account for bulmer - d/Vg = more complicated eq 17
    X = (omegasomething + V_E) / dframe["sigma_sqs"]
    d_over_vg = (3 + X - np.sqrt(1 + 6 * X + X**2)) / 4

    if mode == "directional":
        str_theory_bulmer = str_theory_semibulmer = dframe["dz"] / omegasomething
        theory_str_bm = theory_str_sbm = "Theory"
    else:
        1 / (omegasomething + dframe["sigma_sqs"] + V_E)
        str_theory_semibulmer = 1 / (omegasomething + dframe["sigma_sqs"] * (1 - d_over_vg) + V_E)
        str_theory_bulmer = (1 - d_over_vg) ** 2 / (omegasomething + dframe["sigma_sqs"] * (1 - d_over_vg) + V_E)
        theory_str_bm = "Theory (Bulmer)"
        theory_str_sbm = "Theory (semi-Bulmer)"
    theory_str_bm = theory_str_sbm = "True"

    num_param_vals = np.unique(dframe["x_vars"]).shape[0]
    pts_per_param_val = dframe["x_vars"].shape[0]/num_param_vals

    sig_thresh = 0.05



    #ax = main boxplot (true, gwas)
    #ax2 = S rel err (all)
    #ax3 = w rel err (all)

    #additional plots: z-scores (for significance) vs omega (all)
    #z-score normalized errors? (all)


    # output order is gt_box, gwas_box, Srelerr, wrelerr, zsc_sig, zsc_err
    if smk.group_by_reg_type:
        save_path = Path(smk.input_parquets[-1])
        for regtype in ["true", "wls", "odr"]:
            fig_box, axs_box = plt.subplots(1, 1, figsize=(3.1, 3.1), layout="constrained")
            fig_Srelerr, axs_Srelerr = plt.subplots(1, 1, figsize=(3.1, 3.1), layout="constrained")
            fig_wrelerr, axs_wrelerr = plt.subplots(1, 1, figsize=(3.1, 3.1), layout="constrained")

            fig_zsc_sig, axs_zsc_sig = plt.subplots(1, 1, figsize=(3.1, 3.1), layout="constrained")
            fig_zsc_err, axs_zsc_err = plt.subplots(1, 1, figsize=(3.1, 3.1), layout="constrained")

            axs_zsc_sig.axhline(norm.ppf(sig_thresh / 2), ls="--", lw=1.5, color="r")
            axs_zsc_sig.axhline(-norm.ppf(sig_thresh / 2), ls="--", lw=1.5, color="r")

            S_est = dframe[f"S_ests_{regtype}"]
            S_err = dframe[f"S_errs_{regtype}"]
            str_theory = str_theory_bulmer if regtype=="true" else str_theory_semibulmer
            str_theory = str_theory.to_numpy()
            theory_label = theory_str_bm if regtype=="true" else theory_str_sbm

            box_x_data = []
            box_y_data = []
            box_labels = []

            nonbox_x_data = []
            nonbox_labels = []
            Srelerr_y_data = []
            wrelerr_y_data = []
            zsc_sig_y_data = []
            zsc_err_y_data = []

            if mode == "directional":
                w_plot_y = np.sqrt(dframe["dz"] / (S_est)) / dframe["x_vars"] - 1
            elif "wls" in regtype or "odr" in regtype:
                w_plot_y = np.sqrt(1 / (S_est) - dframe["sigma_sqs"] * (1 - d_over_vg) - V_E) / dframe["x_vars"] - 1
            else:
                w_plot_y = np.sqrt(
                    (1 - d_over_vg) ** 2 / (S_est) - dframe["sigma_sqs"] * (1 - d_over_vg) - V_E) / \
                           dframe["x_vars"] - 1

            if pts_per_param_val > 5:
                box_x_data.extend(dframe["x_vars"].tolist())
                box_y_data.extend(S_est)
                box_labels.extend(dframe["labels"].tolist())

                if np.unique(dframe["x_vars"]).shape[0] == np.unique(str_theory.round(10)).shape[0]:
                    uq_x = np.unique(dframe["x_vars"])
                    log_uq_x = np.log(uq_x)
                    min_x_diff = np.min(np.diff(log_uq_x))
                    _, uq_idxs = np.unique(str_theory.round(10), return_index=True)
                    uq_theory = str_theory[np.sort(uq_idxs)]
                    if uq_theory.shape[0] == 1:
                        uq_theory = np.repeat(uq_theory, uq_x.shape[0])
                    assert uq_theory.shape[0] == uq_x.shape[0]
                    axs_box.hlines(uq_theory, np.exp(log_uq_x - min_x_diff / 4), np.exp(log_uq_x + min_x_diff / 4),
                                  colors="r", lw=1, label=theory_label, ls="--", zorder=3)
                        # if "true" in label:
                        #     box_ax.hlines(str_theory_nothing[np.sort(uq_idxs)], np.exp(log_uq_x - min_x_diff/4), np.exp(log_uq_x+min_x_diff/4), colors="g", lw=1, label="Theory", ls="--", zorder=3)
                        # else:
                        #     box_ax.hlines(str_theory_bulmer[np.sort(uq_idxs)], np.exp(log_uq_x - min_x_diff/4), np.exp(log_uq_x+min_x_diff/4), colors="g", lw=1, label="Theory (Bulmer)", ls="--", zorder=3)
                else:
                    box_x_data.extend(dframe["x_vars"].tolist())
                    box_y_data.extend(str_theory.tolist())
                    box_labels.extend(["Theory"] * str_theory.shape[0])

                nonbox_x_data.extend(dframe["x_vars"].tolist())
                nonbox_labels.extend(dframe["labels"].tolist())
                Srelerr_y_data.extend((S_est / str_theory - 1).tolist())
                wrelerr_y_data.extend(w_plot_y.tolist())
                zsc_sig_y_data.extend((S_est / S_err).tolist())
                zsc_err_y_data.extend(((S_est - str_theory) / S_err).tolist())
            else:
                #this doesn't work - labels need to be separate I think
                axs_box.plot(dframe["x_vars"], str_theory, ".", label="Theory")
                for labeli in smk.labels:
                    dframe_sub = dframe.loc[dframe["labels"]==labeli]
                    axs_box.plot(dframe_sub["x_vars"], S_est[dframe["labels"]==labeli], ".", label=labeli)
                    axs_Srelerr.plot(dframe_sub["x_vars"], (S_est / str_theory - 1)[dframe["labels"]==labeli], ".", label=labeli)
                    axs_wrelerr.plot(dframe_sub["x_vars"], w_plot_y[dframe["labels"]==labeli], ".", label=labeli)
                    axs_zsc_sig.plot(dframe_sub["x_vars"], (S_est / S_err)[dframe["labels"]==labeli], ".", label=labeli)
                    axs_zsc_err.plot(dframe_sub["x_vars"], ((S_est - str_theory) / S_err)[dframe["labels"]==labeli], ".", label=labeli)
            box_df = pdDataFrame(data=zip(box_x_data, box_y_data, box_labels, strict=False),
                                    columns=[x_label, y_label, "Label"])

            snsboxplot(data=box_df, x=x_label, y=y_label, hue="Label", dodge=True, width=.75,
                       ax=axs_box, fliersize=2, boxprops={"lw": 1}, medianprops={"lw": 1}, whiskerprops={"lw": 1},
                       capprops={"lw": 1}, flierprops={"alpha": 1}, native_scale=True, log_scale=[True, False])
            axs_box.set_xlabel(x_label)
            axs_box.set_ylabel(y_label)
            axs_box.set_xticks(np.unique(dframe["x_vars"]))
            axs_box.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axs_box.get_xaxis().minorticks_off()
            axs_box.axhline(0, ls="--", lw=1, color="k")
            # ax.set_yscale("log")
            # axs.set_xscale("log")
            axs_box.legend()
            # if smk.mode == "stabilizing":
            # ax.text(-.18, .96, rf"$\bf{{{subfig_l}}}$", fontsize=13, transform=ax.transAxes)
            fig_box.savefig(save_path.parent / (smk.output_names[0]+f"_{regtype}.pdf"), format="pdf", bbox_inches="tight")
            plt.close(fig_box)
            if pts_per_param_val > 0:
                for ydata, ylabel, axs, fig, num in zip(
                        [Srelerr_y_data, wrelerr_y_data, zsc_sig_y_data, zsc_err_y_data],
                        [r"$\frac{\hat{S}-S}{S}$", r"$\frac{\hat{\omega}-\omega}{\omega}$", "Z-score",
                         r"$\frac{\hat{S} - S}{\hat{se}_S}$"],
                        [axs_Srelerr, axs_wrelerr, axs_zsc_sig, axs_zsc_err],
                        [fig_Srelerr, fig_wrelerr, fig_zsc_sig, fig_zsc_err], [1, 2, 3, 4], strict=False):
                    boxplot_df = pdDataFrame(data=zip(nonbox_x_data, ydata, nonbox_labels, strict=False),
                                     columns=[x_label, ylabel, "Data type"])
                    snsboxplot(data=boxplot_df, x=x_label, y=ylabel, hue="Data type", dodge=True, width=.75,
                               ax=axs, fliersize=2, boxprops={"lw": 1}, medianprops={"lw": 1}, whiskerprops={"lw": 1},
                               capprops={"lw": 1}, flierprops={"alpha": 1})
                    axs.axhline(0, ls="--", lw=1.5, color="k")
                    axs.set_ylabel(ylabel)
                    axs.set_xlabel(x_label)
                    axs.set_xticks(np.unique(dframe["x_vars"]))
                    axs.legend()
                    fig.savefig(save_path.parent / (smk.output_names[num]+f"_{regtype}.pdf"), format="pdf", bbox_inches="tight")
                    plt.close(fig)

    else:
        for ip_i in range(len(smk.input_parquets)):

            label_i = smk.labels[ip_i]
            path_i = Path(smk.input_parquets[ip_i])

            dframe_subset = dframe.loc[dframe["labels"]==label_i]
            dframe_mask = np.where(dframe["labels"]==label_i)[0]

            #extreme abuse of Python
            fig_box_gt, axs_box_gt = plt.subplots(1, 1, figsize=(3.1, 3.1), layout="constrained")
            fig_box_gwas, axs_box_gwas = plt.subplots(1, 1, figsize=(3.1, 3.1), layout="constrained")

            fig_Srelerr, axs_Srelerr = plt.subplots(1, 1, figsize=(3.1, 3.1), layout="constrained")
            fig_wrelerr, axs_wrelerr = plt.subplots(1, 1, figsize=(3.1, 3.1), layout="constrained")

            fig_zsc_sig, axs_zsc_sig = plt.subplots(1, 1, figsize=(3.1, 3.1), layout="constrained")
            fig_zsc_err, axs_zsc_err = plt.subplots(1, 1, figsize=(3.1, 3.1), layout="constrained")

            axs_zsc_sig.axhline(norm.ppf(sig_thresh / 2), ls="--", lw=1.5, color="r")
            axs_zsc_sig.axhline(-norm.ppf(sig_thresh / 2), ls="--", lw=1.5, color="r")

            box_gt_x_data = []
            box_gt_y_data = []
            box_gt_labels = []

            box_gwas_x_data = []
            box_gwas_y_data = []
            box_gwas_labels = []

            nonbox_x_data = []
            nonbox_labels = []
            Srelerr_y_data = []
            wrelerr_y_data = []
            zsc_sig_y_data = []
            zsc_err_y_data = []


            for str_theory, S_est, S_err, box_x, box_y, box_l, box_ax, label, box_tl in (
                    zip([str_theory_bulmer[dframe_mask], str_theory_semibulmer[dframe_mask], str_theory_semibulmer[dframe_mask]],
                        [dframe_subset["S_ests_true"], dframe_subset["S_ests_wls"], dframe_subset["S_ests_odr"]],
                        [dframe_subset["S_errs_true"], dframe_subset["S_errs_wls"], dframe_subset["S_errs_odr"]],
                        [box_gt_x_data, box_gwas_x_data, box_gwas_x_data],
                        [box_gt_y_data, box_gwas_y_data, box_gwas_y_data],
                        [box_gt_labels, box_gwas_labels, box_gwas_labels],
                        [axs_box_gt, axs_box_gwas, axs_box_gwas],
                        [r"Est. (true $\beta$s)", "GWAS-WLS", "GWAS-ODR"],
                        [theory_str_bm, theory_str_sbm, theory_str_sbm], strict=False)):
                # if "WLS" in label:
                #     continue

                if LL_FLAG and "ODR" in label:
                    continue

                if mode == "directional":
                    w_plot_y = np.sqrt(dframe["dz"] / (S_est)) / dframe_subset["x_vars"] - 1
                elif "GWAS" in label:
                    w_plot_y = np.sqrt(1 / (S_est) - dframe_subset["sigma_sqs"] * (1 - d_over_vg) - V_E) / dframe_subset["x_vars"] - 1
                else:
                    w_plot_y = np.sqrt((1 - d_over_vg) ** 2 / (S_est) - dframe_subset["sigma_sqs"] * (1 - d_over_vg) - V_E) / dframe_subset["x_vars"] - 1

                if pts_per_param_val > 10:
                    box_x.extend(dframe_subset["x_vars"].tolist())
                    box_y.extend(S_est)
                    box_l.extend([label]*S_est.shape[0])

                    if np.unique(dframe["x_vars"]).shape[0] == np.unique(str_theory.round(10)).shape[0]:
                        if "ODR" not in label:
                            uq_x = np.unique(dframe_subset["x_vars"])
                            log_uq_x = np.log(uq_x)
                            min_x_diff = np.min(np.diff(log_uq_x))
                            _, uq_idxs = np.unique(str_theory, return_index=True)
                            uq_theory = str_theory[np.sort(uq_idxs)]
                            if uq_theory.shape[0] == 1:
                                uq_theory = np.repeat(uq_theory, uq_x.shape[0])
                            assert uq_theory.shape[0] == uq_x.shape[0]
                            box_ax.hlines(uq_theory, np.exp(log_uq_x - min_x_diff/4), np.exp(log_uq_x+min_x_diff/4), colors="r", lw=1, label=box_tl, ls="--", zorder=3)
                            # if "true" in label:
                            #     box_ax.hlines(str_theory_nothing[np.sort(uq_idxs)], np.exp(log_uq_x - min_x_diff/4), np.exp(log_uq_x+min_x_diff/4), colors="g", lw=1, label="Theory", ls="--", zorder=3)
                            # else:
                            #     box_ax.hlines(str_theory_bulmer[np.sort(uq_idxs)], np.exp(log_uq_x - min_x_diff/4), np.exp(log_uq_x+min_x_diff/4), colors="g", lw=1, label="Theory (Bulmer)", ls="--", zorder=3)
                    else:
                        box_x.extend(dframe_subset["x_vars"].tolist())
                        box_y.extend(str_theory.tolist())
                        box_l.extend(["Theory"]*str_theory.shape[0])

                    nonbox_x_data.extend(dframe_subset["x_vars"].tolist())
                    nonbox_labels.extend([label]*dframe_subset["x_vars"].shape[0])
                    Srelerr_y_data.extend((S_est/str_theory-1).tolist())
                    wrelerr_y_data.extend(w_plot_y.tolist())
                    zsc_sig_y_data.extend((S_est/S_err).tolist())
                    zsc_err_y_data.extend(((S_est-str_theory)/S_err).tolist())
                else:
                    box_ax.plot(dframe_subset["x_vars"], S_est, ".", label=label)
                    if "ODR" not in label:
                        box_ax.plot(dframe_subset["x_vars"], str_theory, ".", label="Theory")
                    axs_Srelerr.plot(dframe_subset["x_vars"], S_est / str_theory - 1, ".", label=label)
                    axs_wrelerr.plot(dframe_subset["x_vars"], w_plot_y, ".", label=label)
                    axs_zsc_sig.plot(dframe_subset["x_vars"], S_est/S_err, ".", label=label)
                    axs_zsc_err.plot(dframe_subset["x_vars"], (S_est-str_theory)/S_err, ".", label=label)

            box_gt_df = pdDataFrame(data=zip(box_gt_x_data, box_gt_y_data, box_gt_labels, strict=False), columns=[x_label, y_label, "Data type"])
            box_gwas_df = pdDataFrame(data=zip(box_gwas_x_data, box_gwas_y_data, box_gwas_labels, strict=False), columns=[x_label, y_label, "Data type"])
            # if smk.mode == "directional":
            #     box_gwas_df = pdconcat([box_gt_df, box_gwas_df])
            for df, ax, fig, num, _subfig_l in zip([box_gt_df, box_gwas_df],
                                                  [axs_box_gt, axs_box_gwas],
                                                  [fig_box_gt, fig_box_gwas],
                                                  [0,1],
                                                    smk.boxplot_letters, strict=False):
                # if smk.mode == "stabilizing" and num == 1:
                #     plt.rcParams["axes.prop_cycle"] = cycler(color=colorlist[1:])
                snsboxplot(data=df, x=x_label, y=y_label, hue="Data type", dodge=True, width=.3,
                           ax=ax, fliersize=2, boxprops={"lw": 1}, medianprops={"lw": 1}, whiskerprops={"lw": 1},
                           capprops={"lw": 1}, flierprops={"alpha": 1}, native_scale=True, log_scale=[True, False])
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_xticks(np.unique(dframe_subset["x_vars"]))
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.get_xaxis().minorticks_off()
                ax.axhline(0, ls="--", lw=1, color="k")
                #ax.set_yscale("log")
                #axs.set_xscale("log")
                ax.legend()
                #if smk.mode == "stabilizing":
                #ax.text(-.18, .96, rf"$\bf{{{subfig_l}}}$", fontsize=13, transform=ax.transAxes)
                fig.savefig(path_i.parent / smk.output_names[num], format="pdf", bbox_inches="tight")
                plt.close(fig)
            if pts_per_param_val > 0:
                for ydata, ylabel, axs, fig, num in zip([Srelerr_y_data,wrelerr_y_data, zsc_sig_y_data, zsc_err_y_data],
                                    [r"$\frac{\hat{S}-S}{S}$", r"$\frac{\hat{\omega}-\omega}{\omega}$", "Z-score", r"$\frac{\hat{S} - S}{\hat{se}_S}$"],
                                    [axs_Srelerr, axs_wrelerr, axs_zsc_sig, axs_zsc_err], [fig_Srelerr, fig_wrelerr, fig_zsc_sig, fig_zsc_err], [2,3,4,5], strict=False):
                    plot_df = pdDataFrame(data=zip(nonbox_x_data, ydata, nonbox_labels, strict=False), columns=[x_label, ylabel, "Data type"])
                    snsboxplot(data=plot_df, x=x_label, y=ylabel, hue="Data type", dodge=True, width=.3,
                               ax=axs, fliersize=2, boxprops={"lw": 1}, medianprops={"lw": 1}, whiskerprops={"lw": 1},
                               capprops={"lw": 1}, flierprops={"alpha": 1})
                    axs.axhline(0, ls="--", lw=1.5, color="k")
                    axs.set_ylabel(ylabel)
                    axs.set_xlabel(x_label)
                    axs.set_xticks(np.unique(dframe_subset["x_vars"]))
                    axs.legend()
                    fig.savefig(path_i.parent / smk.output_names[num], format="pdf", bbox_inches="tight")
                    plt.close(fig)


if __name__ == "__main__":
    main()
