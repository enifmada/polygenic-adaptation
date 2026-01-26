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

base_str = "GWASDIR"

dz = 0.005
omega = 0.4

base_dir = Path("../../../polyoutput/slim_testing/")
sample_times = np.loadtxt(base_dir / "GB_v54.1_capture_only_sample_sizes_fixed.table", skiprows=1)
fig, axs = plt.subplots(1, 1, figsize=(3.1, 1.5), layout="constrained")
axs.invert_xaxis()
axs.bar(150 - sample_times[:, 0], sample_times[:, 1])
axs.set_ylabel("# of samples")
axs.set_xlabel("Generations before present")
# axs[0].text(-.24, .92, r"$\bf{B}$", fontsize=13, transform=axs[0].transAxes)

fig.savefig(base_dir / "pressamples.pdf", format="pdf", bbox_inches="tight")
