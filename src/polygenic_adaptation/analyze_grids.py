import numpy as np
import pandas as pd
import json
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from pathlib import Path

with open(snakemake.input[-1], "r") as file:
    jps = json.load(file)

d_hits = 0
d_miss = 0
s_hits = 0
s_miss = 0
direc_estimates = []
stab_estimates = []
for grid_file in snakemake.input[:-1]:
    grid = np.loadtxt(grid_file)
    raw_grid = grid[0, :]
    dll_vals = grid[1::2, :]
    sll_vals = grid[2::2, :]

    expanded_x = np.linspace(-10, 10, 100000)
    summed_dlls = np.zeros_like(expanded_x)
    summed_slls = np.zeros_like(expanded_x)
    for loc in range(dll_vals.shape[0]):

        #stand-in for individual beta values because we don't have those yet

        #*2 b/c conversion from s2 = s to s1 = s
        sdz_est_grid = raw_grid/(2*jps["beta_values"])
        #/2 b/c I think that's the right coefficient in the actual equations?
        s_est_grid = raw_grid/(jps["beta_values"]**2/2)

        dll_spline = CubicSpline(sdz_est_grid, dll_vals[loc, :])
        sll_spline = CubicSpline(s_est_grid, sll_vals[loc, :])

        dll_ests = dll_spline(expanded_x)
        summed_dlls += dll_ests

        sll_ests = sll_spline(expanded_x)
        summed_slls += sll_ests

    dll_max = np.max(summed_dlls)
    dll_argmax = expanded_x[np.argmax(summed_dlls)]

    sll_max = np.max(summed_slls)
    sll_argmax = expanded_x[np.argmax(summed_slls)]
    print(f"{Path(grid_file).name}: {dll_argmax:.4f} {sll_argmax:.4f}")

    if "dz0.0_" in grid_file:
        #stabilizing only
        stab_estimates.append(-sll_argmax)

        if dll_max > sll_max:
            s_miss += 1
        else:
            s_hits += 1

    else:
        #directional only
        direc_estimates.append(dll_argmax/.1)

        if dll_max > sll_max:
            d_hits += 1
        else:
            d_miss += 1
d = {"Inf Direc": [d_hits, s_miss], "Inf Stab": [d_miss, s_hits]}

ctable = pd.DataFrame(data=d, index=["Sim Direc", "Sim Stab"])

ctable.to_csv(snakemake.output[0])

fig, axs = plt.subplots(1,1,figsize=(3.1, 3.1), layout="constrained")

axs.boxplot([direc_estimates, stab_estimates], labels=["directional", "stabilizing"])
fig.savefig(snakemake.output[1], format="pdf", bbox_inches="tight")
plt.close(fig)


