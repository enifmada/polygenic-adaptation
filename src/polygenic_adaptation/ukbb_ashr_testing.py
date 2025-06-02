from __future__ import annotations

import time

import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr

col_names = ["PosteriorMean", "PosteriorSD"]

fname = "blood_EOSINOPHIL_COUNT_ashvalues.csv.gz"
data_array = pd.read_csv(fname)
pandas2ri.activate()
ashr = importr('ashr')
r_df = pandas2ri.py2rpy(data_array)
start = time.time()
ashres = ashr.ash(r_df.rx2("Beta"), r_df.rx2("se"), mixcompdist="normal")
ashres_res = ashres.rx2("result")
ash_cols_needed = ashres_res.rx(True, r.c("PosteriorMean", "PosteriorSD"))
cols = [list(r['as.numeric'](ash_cols_needed.rx2(i+1))) for i in range(len(col_names))]
ashres_np = np.column_stack(cols)
data_array[["ash_beta", "ash_se"]] = ashres_np
data_array.to_csv()

#r["saveRDS"](ashres, "ash_output_test.rds")
