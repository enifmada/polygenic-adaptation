from __future__ import annotations

import argparse
import sys
from itertools import tee as ittee
from json import dump as jdump
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.append(str(Path(__file__).resolve().parent.parent))
from tqdm import tqdm

from polygenic_adaptation.hmm_core import HMM


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
    parser.add_argument(
        "--sid_dict", nargs="*", default="", help="initial condition dictionary"
    )
    parser.add_argument(
        "-Ne", type=int, default=10000, help="effective population size for the HMM"
    )
    parser.add_argument(
        "--progressbar", action="store_true", help="adds a tqdm progress bar"
    )
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

    MIN_GRID_VAL = 1e-8
    pos_grid = np.geomspace(
        MIN_GRID_VAL, args_dict["grid_s_max"], args_dict["num_half_grid_points"]
    )
    full_grid = np.concatenate((-pos_grid[::-1], [0], pos_grid))

    interp_grid = np.linspace(-args_dict["grid_s_max"], args_dict["grid_s_max"], 1001)
    direc_unif_lls = np.zeros((data_matrix.shape[0], full_grid.shape[0]))
    stab_unif_lls = np.zeros((data_matrix.shape[0], full_grid.shape[0]))
    direc_EM_lls = np.zeros_like(direc_unif_lls)
    stab_EM_lls = np.zeros_like(stab_unif_lls)

    # init_ests = np.zeros((data_matrix.shape[0], args_dict["hidden_states"]))
    # neutral_a = hmm.calc_transition_probs_old([0, 0])
    # for i in tqdm(range(data_matrix.shape[0])):
    #     init_ests[i, :], _ = hmm.compute_one_init_est(data_matrix[i, 2::3], data_matrix[i, 1::3],
    #                                                             data_matrix[i, ::3], neutral_a, tol=1e-3, max_iter=1000,
    #                                                             min_init_val=1e-8, )
    enum_object = (
        enumerate(tqdm(full_grid)) if args_dict["progressbar"] else enumerate(full_grid)
    )
    loop_1, loop_2 = ittee(enum_object, 2)
    for s_i, s in loop_1:
        direc_unif_lls[:, s_i] = hmm.compute_multiple_ll(
            s1=s / 2, s2=s, data_matrix=data_matrix, init_states=None
        )
        stab_unif_lls[:, s_i] = hmm.compute_multiple_ll(
            s1=s, s2=0, data_matrix=data_matrix, init_states=None
        )
        # direc_EM_lls[:, s_i] = hmm.compute_multiple_ll(s1=s/2,s2=s,data_matrix=data_matrix,init_states=init_ests)
        # stab_EM_lls[:, s_i] = hmm.compute_multiple_ll(s1=s,s2=0,data_matrix=data_matrix,init_states=init_ests)

    direc_init_ests = np.zeros((data_matrix.shape[0], args_dict["hidden_states"]))
    stab_init_ests = np.zeros_like(direc_init_ests)
    for i in tqdm(range(data_matrix.shape[0])):
        direc_pchip = PchipInterpolator(full_grid, direc_unif_lls[i, :])
        stab_pchip = PchipInterpolator(full_grid, stab_unif_lls[i, :])
        direc_s = interp_grid[np.argmax(direc_pchip(interp_grid))]
        direc_a = hmm.calc_transition_probs_old([direc_s / 2, direc_s])
        direc_init_ests[i, :], _ = hmm.compute_one_init_est(
            data_matrix[i, 2::3],
            data_matrix[i, 1::3],
            data_matrix[i, ::3],
            direc_a,
            tol=1e-3,
            max_iter=1000,
            min_init_val=1e-8,
        )
        stab_s = interp_grid[np.argmax(stab_pchip(interp_grid))]
        stab_a = hmm.calc_transition_probs_old([stab_s, 0])
        stab_init_ests[i, :], _ = hmm.compute_one_init_est(
            data_matrix[i, 2::3],
            data_matrix[i, 1::3],
            data_matrix[i, ::3],
            stab_a,
            tol=1e-3,
            max_iter=1000,
            min_init_val=1e-8,
        )

    for s_i, s in loop_2:
        direc_EM_lls[:, s_i] = hmm.compute_multiple_ll(
            s1=s / 2, s2=s, data_matrix=data_matrix, init_states=direc_init_ests
        )
        stab_EM_lls[:, s_i] = hmm.compute_multiple_ll(
            s1=s, s2=0, data_matrix=data_matrix, init_states=stab_init_ests
        )
    combined_grid = np.zeros((4 * direc_unif_lls.shape[0] + 1, direc_unif_lls.shape[1]))
    combined_grid[0, :] = full_grid
    for row in range(direc_unif_lls.shape[0]):
        combined_grid[4 * row + 1, :] = direc_unif_lls[row, :]
        combined_grid[4 * row + 2, :] = stab_unif_lls[row, :]
        combined_grid[4 * row + 3, :] = direc_EM_lls[row, :]
        combined_grid[4 * row + 4, :] = stab_EM_lls[row, :]
    np.savetxt(
        args_dict["output_path"],
        combined_grid,
        header="s_grid followed by direc_unif_ll, stab_unif_ll, direc_EM_ll, stab_EM_ll for each rep",
    )
    if "snakemake" not in args_dict or not args_dict["snakemake"]:
        json_fname = f"{Path(args_dict['output_path']).with_suffix('')}_params.json"
        with Path(json_fname).open("w", encoding="locale") as file:
            jdump(args_dict, file)


if __name__ == "__main__":
    main()
