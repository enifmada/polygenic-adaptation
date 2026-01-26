from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np

# def main():
#     parser = ArgumentParser()
#     parser.add_argument("--base_dir", type=str, help="beta mode")
#     parser.add_argument("--omega_sq", type=float, help="omega squared value")
#     parser.add_argument("--num_sims", type=int, help="freq mode")
#
#
# if __name__ == "__main__":
#     main()


for num_loci in [500, 1000, 2000, 4000, 8000]:
    base_dir = Path(f"../../../polyoutput/slim_testing/vgmatch{num_loci}/slims")
    for s in range(16):
        for omega in [0.1, 0.2, 0.4, 0.8, 1.6]:
            gfile = base_dir/f"vgmatch{num_loci}_w{omega}_s{s}_allgenos.vcf"
            pfile = base_dir/f"vgmatch{num_loci}_w{omega}_s{s}_phenotypes.txt"
            sfile = base_dir/f"vgmatch{num_loci}_w{omega}_s{s}_slim.txt"

            gdata = np.loadtxt(gfile)
            np.savez_compressed(gfile.with_suffix(".npz"), gdata)

            pdata = np.loadtxt(pfile)
            np.savez_compressed(pfile.with_suffix(".npz"), pdata)

            sdata = np.loadtxt(sfile)
            np.savez_compressed(sfile.with_suffix(".npz"), sdata)