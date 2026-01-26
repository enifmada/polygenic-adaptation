from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# def main():
#     parser = ArgumentParser()
#     parser.add_argument("--base_dir", type=str, help="beta mode")
#     parser.add_argument("--omega_sq", type=float, help="omega squared value")
#     parser.add_argument("--num_sims", type=int, help="freq mode")
#
#
# if __name__ == "__main__":
#     main()

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

for id_str, _label, suffix in zip(["POLYFULL", "GWASFULL"], ["B", "A"], ["polystab", "stab"], strict=False):

    if suffix == "stab":
        temp_c = "#1D6996"
        colorlist[0] = colorlist[-1]
        colorlist[-1] = temp_c
        plt.rcParams["axes.prop_cycle"] = cycler(color=colorlist)

    base_dir = Path(f"../../../polyoutput/slim_testing/{id_str}")
    all_betas = np.array([0])
    true_beta = 0.0075
    num_reps = 32
    num_loci = 500
    p_init = 0.2
    omega = 0.2
    for i in range(num_reps):
        gwas_betas = np.loadtxt(base_dir/f"betas/{id_str}_w{omega}_s{i}_betas_gwas.txt")[:, 0]
        all_betas = np.concatenate((all_betas, gwas_betas))
    all_betas = all_betas[1:]

    pos_mean = np.mean(all_betas[all_betas >= 0])
    neg_mean = np.mean(all_betas[all_betas < 0])

    fig, axs = plt.subplots(1,1, figsize=(3.1, 2), layout="constrained")
    axs.hist(all_betas, bins="auto", alpha=0.5)

    axs.axvline(pos_mean, c="red", label="Empirical mean")
    axs.axvline(neg_mean, c="red")

    axs.axvline(true_beta, ls = "--", c="black", label=r"True $\beta$")
    axs.axvline(-true_beta, ls = "--", c="black")

    sigma_sq = num_loci * 2 * p_init * (1-p_init) * true_beta**2
    X = omega**2/sigma_sq
    d_over_vg = (3 + X - np.sqrt(1 + 6 * X + X**2)) / 4

    axs.axvline(true_beta * (1-d_over_vg), c="green", ls="-.", label=r"Bulmer $\beta$")
    axs.axvline(-true_beta * (1-d_over_vg), c="green", ls="-.")


    axs.set_xlabel("Effect size")
    axs.set_ylabel("Counts")
    axs.legend()
    #axs.set_xticks([-true_beta, 0, true_beta])

    #axs.text(-.24, .92, rf"$\bf{{{label}}}$", fontsize=13, transform=axs.transAxes)

    fig.savefig(base_dir/f"GWAS_hist_pres{suffix}.pdf", format="pdf", bbox_inches="tight")