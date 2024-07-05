from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from tqdm import tqdm

from polygenic_adaptation.hmm_core import HMM

data_file = "../../../polyoutput/sims/data/normal_betas_g300_S1.0_dz0.1_uniform_rep7_data.csv"

data_array = np.loadtxt(data_file, dtype=int)

weird_vals = np.array([35,63])

s1_grid = np.linspace(-.1, .1, 21)
s2_grid = np.copy(s1_grid)

lls = np.zeros((s1_grid.shape[0], s1_grid.shape[0], 2))

detective_hmm = HMM(num_approx=500, Ne=10000, init_cond="uniform")
for s1_i, s1 in enumerate(tqdm(s1_grid)):
    for s2_i, s2 in enumerate(s2_grid):
        lls[s1_i, s2_i, :] = detective_hmm.compute_multiple_ll(s1, s2, data_array[weird_vals, :])



