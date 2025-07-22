from __future__ import annotations

import argparse
import sys
from json import dump as jdump
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.append(str(Path(__file__).resolve().parent.parent))
from tqdm import tqdm

from polygenic_adaptation.hmm_core import HMM


def compute_ll_wrapper(hmm, s, data_matrix):
    direc_res = hmm.compute_multiple_ll(s / 2, s, data_matrix)
    stab_res = hmm.compute_multiple_ll(s, 0, data_matrix)
    return direc_res, stab_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="path to input dataset")
    parser.add_argument(
        "output_path",
        type=str,
        help="path to output csv - additional files may be created in the same directory",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--time_before_present",
        action="store_true",
        help="dates provided start at a number at the earliest time and count down towards the present",
    )
    group.add_argument(
        "--time_after_zero",
        action="store_true",
        help="dates provided start at zero at the earliest time and count up towards the present",
    )
    parser.add_argument(
        "--grid_s_max",
        type=float,
        default=0.25,
        help="maximum s value to use in the grid",
    )
    parser.add_argument(
        "-np",
        "--num_half_grid_points",
        type=int,
        default=15,
        help="number of half-grid points to use - the full grid will have 2*np + 1 points",
    )
    parser.add_argument(
        "-ytg",
        "--years_to_gen",
        type=float,
        default=1,
        help="years per generation in VCF or CSV",
    )
    parser.add_argument(
        "-hs",
        "--hidden_states",
        type=int,
        help="number of approx states in HMM",
        default=500,
    )
    parser.add_argument(
        "-sid",
        "--starting_init_dist",
        default="uniform",
        help="initial initial condition to use",
    )
    parser.add_argument("--sid_dict", nargs="*", default="", help="initial condition dictionary")
    parser.add_argument("-Ne", type=int, default=10000, help="effective population size for the HMM")
    parser.add_argument("--progressbar", action="store_true", help="adds a tqdm progress bar")
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="if inputting a VCF, save a CSV to future reduce pre-processing time",
    )
    parser.add_argument(
        "--info_file",
        type=argparse.FileType("rb"),
        help="sample times file (if input = VCF)",
    )
    parser.add_argument(
        "--info_cols",
        type=str,
        nargs=2,
        default=["Genetic_ID", "Date_mean"],
        help="names of the ID and dates columns in the sample times file (if input = VCF)",
    )
    parser.add_argument(
        "--full_output",
        action="store_true",
        help="save a pickle file with a full set of outputs (in addition to the CSV)",
    )
    parser.add_argument(
        "-nc",
        "--num_cores",
        type=int,
        default=1,
        help="number of CPU cores to parallelize over",
    )
    parser.add_argument(
        "--force",
        type=str,
        nargs=1,
        help="if the VCF file only contains homozygous loci, force it to be read as either haploid or diploid",
    )
    parser.add_argument(
        "--snakemake",
        action="store_true",
        help="whether or not this script was run as part of a snakemake workflow. If so, do not save the params as a json because params.json already exists.",
    )
    args_dict = vars(parser.parse_args())
    actual_sid_dict = {}

    if args_dict["sid_dict"] is not None:
        for ic_pair in args_dict["sid_dict"]:
            k, v = ic_pair.split("=")
            try:
                actual_sid_dict[k] = float(v)
            except ValueError:
                actual_sid_dict[k] = v
    args_dict["sid_dict"] = actual_sid_dict
    hmm = HMM(
        args_dict["hidden_states"],
        args_dict["Ne"],
        args_dict["starting_init_dist"],
        **args_dict["sid_dict"],
    )

    if Path(args_dict["input_path"]).suffix == ".csv":
        data_matrix = np.loadtxt(args_dict["input_path"], dtype=int)
    else:
        # equivalent of pass but the thing exists
        data_matrix = np.zeros((1,))

    MIN_GRID_VAL = 5e-5
    pos_grid = np.geomspace(MIN_GRID_VAL, args_dict["grid_s_max"], args_dict["num_half_grid_points"])
    full_grid = np.concatenate((-pos_grid[::-1], [0], pos_grid))

    np.linspace(-args_dict["grid_s_max"], args_dict["grid_s_max"], 1001)
    direc_unif_lls = np.zeros((data_matrix.shape[0], full_grid.shape[0]))
    stab_unif_lls = np.zeros((data_matrix.shape[0], full_grid.shape[0]))

    if args_dict["num_cores"] > 1:
        parallel_loop = tqdm(full_grid) if args_dict["progressbar"] else full_grid
        with Parallel(n_jobs=args_dict["num_cores"]) as parallel:
            res = parallel(delayed(compute_ll_wrapper)(hmm, p_s, data_matrix) for p_s in parallel_loop)
        direc_lls = [rp[0] for rp in res]
        stab_lls = [rp[1] for rp in res]
        direc_unif_lls[:, :] = np.array(direc_lls).T.squeeze()
        stab_unif_lls[:, :] = np.array(stab_lls).T.squeeze()
    else:
        for s_i, s in enumerate(tqdm(full_grid)) if args_dict["progressbar"] else enumerate(full_grid):
            direc_unif_lls[:, s_i] = hmm.compute_multiple_ll(s1=s / 2, s2=s, data_matrix=data_matrix)
            stab_unif_lls[:, s_i] = hmm.compute_multiple_ll(s1=s, s2=0, data_matrix=data_matrix)

    combined_grid = np.zeros((2 * direc_unif_lls.shape[0] + 1, direc_unif_lls.shape[1]))
    combined_grid[0, :] = full_grid
    for row in range(direc_unif_lls.shape[0]):
        combined_grid[2 * row + 1, :] = direc_unif_lls[row, :]
        combined_grid[2 * row + 2, :] = stab_unif_lls[row, :]
    np.savetxt(
        args_dict["output_path"],
        combined_grid,
        header="s_grid followed by direc_unif_ll+stab_unif_ll for each rep",
    )
    if "snakemake" not in args_dict or not args_dict["snakemake"]:
        json_fname = f"{Path(args_dict['output_path']).with_suffix('')}_params.json"
        with Path(json_fname).open("w", encoding="locale") as file:
            jdump(args_dict, file)


if __name__ == "__main__":
    main()
