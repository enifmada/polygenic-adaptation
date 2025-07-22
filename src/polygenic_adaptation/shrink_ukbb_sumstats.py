from __future__ import annotations

import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr

ash_col_names = ["PosteriorMean", "PosteriorSD", "qvalue"]
ashg_col_names = ["pi", "sd"]


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", nargs="*", help="input")
    parser.add_argument("-o", "--output", nargs="*", help="output")
    smk = parser.parse_args()
    data_array = pd.read_csv(smk.input[0], sep="\t")
    pandas2ri.activate()
    ashr = importr("ashr")
    r_df = pandas2ri.py2rpy(data_array)
    time.time()
    ashres = ashr.ash(r_df.rx2("Beta"), r_df.rx2("se"), mixcompdist="normal")
    ash_gs = ashres.rx2("fitted_g")
    ashres_res = ashres.rx2("result")
    ash_cols_needed = ashres_res.rx(True, r.c("PosteriorMean", "PosteriorSD", "qvalue"))
    cols = [list(r["as.numeric"](ash_cols_needed.rx2(i + 1))) for i in range(len(ash_col_names))]
    ashres_np = np.column_stack(cols)
    pi_0 = np.max(ashres_np[:, -1])
    m_0 = ashres_np.shape[0] * pi_0
    t_args = np.argsort(ashres_np[:, -1])
    ranks = np.empty_like(t_args)
    ranks[t_args] = np.arange(t_args.shape[0])
    p_vals = ashres_np[:, -1] * ranks / m_0
    ashres_pd = pd.DataFrame(ashres_np[:, :-1], columns=["ash_beta", "ash_se"])
    ashres_pd["ash_p"] = p_vals
    ashres_pd.to_csv(smk.output[0], compression="gzip", index=False)
    # r["saveRDS"](ashres, "ash_output_test.rds")
    ashg_pi = ash_gs.rx2("pi")
    ashg_sd = ash_gs.rx2("sd")
    ashg_np = np.hstack((np.array(ashg_pi), np.array(ashg_sd)))

    np.savetxt(smk.output[1], ashg_np)


if __name__ == "__main__":
    main()
