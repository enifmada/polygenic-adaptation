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

base_str = "FIG1-STAB"

dz = 0.01
omega = 0.4

base_dir = Path(f"../../../polyoutput/slim_testing/{base_str}")
slim_out = np.loadtxt(base_dir/f"slims/{base_str}_w{omega}_s{0}_slim.txt", skiprows=1).T
slim_freqs = slim_out[3:, :]

fig, axs = plt.subplots(1,1, figsize=(3.1, 1.5), layout="constrained")
# axs[1].invert_xaxis()
# axs[1].plot(slim_freqs.T, alpha = 0.05, lw=0.2, color="k")
# # random_idxs = np.random.default_rng(6).choice(np.arange(slim_freqs.shape[0]), size=10)
# # for random_idx in random_idxs:
# #     axs[1].plot(slim_freqs[random_idx])
#
# axs[1].set_xlabel("Generation")
# axs[1].set_ylabel("Frequency")
#axs[1].set_ylim([0,1])
#axs[1].set_xlim([0, 250])
#axs.set_xticks([-true_beta, 0, true_beta])

axs.plot(slim_out[1, :]/omega**2)
axs.set_ylabel("Trait value")
axs.set_xlim([0, 40])
axs.set_xlabel("Generation")
#axs[0].set_ylim([0,2.2])
#axs[0].text(-.24, .92, r"$\bf{B}$", fontsize=13, transform=axs[0].transAxes)

fig.savefig(base_dir/f"{base_str}_prestraitplot_spooky.pdf", format="pdf", bbox_inches="tight")