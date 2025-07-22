from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from rpy2.robjects import r

col_names = ["betahat", "sebatahat", "PosteriorMean", "PosteriorSD", "svalue", "qvalue"]
ashres = r["readRDS"]("ash_output_test.rds")


ashres_res = ashres.rx2("result")
ash_cols_needed = ashres_res.rx(
    True,
    r.c("betahat", "sebetahat", "PosteriorMean", "PosteriorSD", "svalue", "qvalue"),
)

cols = [list(r["as.numeric"](ash_cols_needed.rx2(i + 1))) for i in range(len(col_names))]

ashres_np = np.column_stack(cols)

# print(ashres_np.shape)

# data_array = pd.read_csv("blood_EOSINOPHIL_COUNT_ashvalues.csv.gz")

# fig, axs = plt.subplots(1,1,figsize=(5,5))
# axs.plot(data_array["Beta"][:1000], ashres_np[:1000, 0], "b.")
# fig.savefig("betas.png", format="png")
#
# fig2, axs2 = plt.subplots(1,1,figsize=(5,5))
# axs2.plot(data_array["se"][:1000], ashres_np[:1000, 1], "b.")
# fig2.savefig("se.png", format="png")
#
# fig3, axs3 = plt.subplots(1,1,figsize=(5,5))
# axs3.plot(data_array["Beta"][:1000], ashres_np[:1000, 2], "b.")
# fig3.savefig("betas_posterior.png", format="png")
#
# fig4, axs4 = plt.subplots(1,1,figsize=(5,5))
# axs4.plot(data_array["se"][:1000], ashres_np[:1000, 3], "b.")
# fig4.savefig("se_posterior.png", format="png")

fig5, axs5 = plt.subplots(1, 1, figsize=(5, 5))
axs5.hist(ashres_np[:, 4], bins="auto")
fig5.savefig("sval_hist.png")


# some dude on reddit's weird conversion
pi_0 = np.max(ashres_np[:, 5])
m_0 = ashres_np.shape[0] * pi_0
t_args = np.argsort(ashres_np[:, 5])
ranks = np.empty_like(t_args)
ranks[t_args] = np.arange(t_args.shape[0])
p_vals = ashres_np[:, 5] * ranks / m_0

fig6, axs6 = plt.subplots(1, 1, figsize=(5, 5))
axs6.hist(p_vals, bins="auto")
fig6.savefig("conv_qval_hist.png")

svals_to_p = np.power(2, -ashres_np[:, 4])
fig7, axs7 = plt.subplots(1, 1, figsize=(5, 5))
axs7.plot(svals_to_p[:10000], p_vals[:10000], "b.")
fig7.savefig("s_vs_q.png")
