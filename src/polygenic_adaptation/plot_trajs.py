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

for sel_type, _label in zip(["STAB", "DIR"], ["A", "B"], strict=False):
    base_str = f"FIG1-{sel_type}"

    base_dir = Path(f"../../../polyoutput/slim_testing/{base_str}")
    slim_out = np.loadtxt(base_dir / f"slims/{base_str}_w0.1_s{0}_slim.txt", skiprows=1).T
    tbetas = np.loadtxt(base_dir / f"betas/{base_str}_w0.1_s{0}_betas.txt")
    slim_freqs = slim_out[3:, :]

    fig, axs = plt.subplots(1, 1, figsize=(3.1, 3.1), layout="constrained")
    axs.plot(slim_freqs[tbetas >= 0].T, alpha=0.3, color=colorlist[0])
    axs.plot(slim_freqs[tbetas < 0].T, alpha=0.3, color=colorlist[3])
    if sel_type == "DIR":
        axs.plot(slim_freqs[tbetas < 0].T, color=colorlist[3])
    else:
        axs.plot(slim_freqs[((slim_freqs[:, 0] > 0.5) & (tbetas >= 0))].T, color=colorlist[0])
        axs.plot(slim_freqs[((slim_freqs[:, 0] > 0.5) & (tbetas < 0))].T, color=colorlist[3])
    axs.set_xlabel("Generation")
    axs.set_ylabel("Frequency")
    axs.set_ylim([0, 1])
    axs.set_xlim([0, 125])
    # axs.set_xticks([-true_beta, 0, true_beta])

    # axs.text(-.24, .92, rf"$\bf{{{label}}}$", fontsize=13, transform=axs.transAxes)

    fig.savefig(base_dir / f"{base_str}_prestrajplot_partial2.pdf", format="pdf", bbox_inches="tight")
