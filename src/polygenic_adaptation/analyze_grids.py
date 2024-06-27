from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator
from tqdm import tqdm

# from polyutil import snakemake

# snakemake = snakemake(["../../../polyoutput/sims/analysis/normal_betas_g300_S1.0_dz0.0_uniform_rep0_grid.csv", "../../../polyoutput/sims/data/normal_betas_g300_S1.0_dz0.0_uniform_rep0_betas.txt", "../../simulation_params/normal_sim_params.json"], ["../../../polyoutput/sims/analysis/normal_betas_confusion_table.csv", "../../../polyoutput/sims/plots/normal_betas_estimated_gradient.pdf"])

jps = json.loads(Path(snakemake.input[-1]).read_text())  # noqa: F821

# MIN_BETA_VALUE = 1e-5
d_hits = 0
d_miss = 0
s_hits = 0
s_miss = 0
direc_estimates = []
stab_estimates = []
num_inputs = len(snakemake.input)  # noqa: F821
for grid_i in tqdm(range((num_inputs - 1) // 2)):
    assert (
        Path(snakemake.input[grid_i]).name.rpartition("_")[0]  # noqa: F821
        == Path(snakemake.input[grid_i + (num_inputs - 1) // 2]).name.rpartition("_")[0]  # noqa: F821
    )
    grid = np.loadtxt(snakemake.input[grid_i])  # noqa: F821
    betas = np.loadtxt(snakemake.input[grid_i + (num_inputs - 1) // 2])  # noqa: F821
    raw_grid = grid[0, :]
    dll_vals = grid[1::2, :]
    sll_vals = grid[2::2, :]

    max_signed_beta = np.max(np.abs(betas))
    expanded_direc_x = np.linspace(
        raw_grid[0] / (2 * max_signed_beta),
        raw_grid[-1] / (2 * max_signed_beta),
        100000,
    )
    expanded_stab_x = np.linspace(
        raw_grid[0] / (max_signed_beta**2 / 2),
        raw_grid[-1] / (max_signed_beta**2 / 2),
        100000,
    )
    summed_dlls = np.zeros_like(expanded_direc_x)
    summed_slls = np.zeros_like(expanded_stab_x)
    all_dll_ests = np.zeros((dll_vals.shape[0], summed_dlls.shape[0]))
    all_sll_ests = np.zeros((dll_vals.shape[0], summed_dlls.shape[0]))
    for loc in range(dll_vals.shape[0]):
        # stand-in for individual beta values because we don't have those yet

        # *2 b/c conversion from s2 = s to s1 = s
        sdz_est_grid = raw_grid / (2 * betas[loc])
        # /2 b/c I think that's the right coefficient in the actual equations?
        s_est_grid = raw_grid / (betas[loc] ** 2 / 2)
        sll_spline = PchipInterpolator(s_est_grid, sll_vals[loc, :])
        if betas[loc] >= 0:
            dll_spline = CubicSpline(sdz_est_grid, dll_vals[loc, :])
            # dll_ests = np.interp(expanded_direc_x, sdz_est_grid, dll_vals[loc, :])
        else:
            dll_spline = CubicSpline(sdz_est_grid[::-1], dll_vals[loc, ::-1])
            # dll_ests = np.interp(expanded_direc_x, sdz_est_grid[::-1], dll_vals[loc, ::-1])

        dll_ests = dll_spline(expanded_direc_x)
        all_dll_ests[loc, :] = dll_ests
        summed_dlls += dll_ests

        sll_ests = np.interp(expanded_stab_x, s_est_grid, sll_vals[loc, :])
        sll_ests = sll_spline(expanded_stab_x)
        all_sll_ests[loc, :] = sll_ests
        summed_slls += sll_ests

    fig, axs = plt.subplots(2, 1, figsize=(5, 10), layout="constrained")
    for loc in range(dll_vals.shape[0]):
        axs[0].plot(expanded_direc_x, all_dll_ests[loc, :])
        axs[1].plot(expanded_stab_x, all_sll_ests[loc, :])
    fig.suptitle(f"{grid_i}")
    fig.savefig(
        Path(snakemake.output[1]).parent / f"{grid_i}_all_pchip_ests.pdf",  # noqa: F821
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    dll_max = np.max(summed_dlls)
    dll_argmax = expanded_direc_x[np.argmax(summed_dlls)]

    sll_max = np.max(summed_slls)
    sll_argmax = expanded_stab_x[np.argmax(summed_slls)]

    if "dz0.0_" in snakemake.input[grid_i]:  # noqa: F821
        # stabilizing only
        stab_estimates.append(-sll_argmax)

        if dll_max > sll_max:
            s_miss += 1
        else:
            s_hits += 1

    else:
        # directional only
        direc_estimates.append(dll_argmax)

        if dll_max > sll_max:
            d_hits += 1
        else:
            d_miss += 1
d = {"Inf Direc": [d_hits, s_miss], "Inf Stab": [d_miss, s_hits]}

ctable = pd.DataFrame(data=d, index=["Sim Direc", "Sim Stab"])

ctable.to_csv(snakemake.output[0])  # noqa: F821

fig, axs = plt.subplots(1, 2, figsize=(3.1, 3.1), layout="constrained")

axs[0].boxplot([direc_estimates], labels=["directional"])
axs[1].boxplot([stab_estimates], labels=["stabilizing"])
fig.savefig(snakemake.output[1], format="pdf", bbox_inches="tight")  # noqa: F821
plt.close(fig)
