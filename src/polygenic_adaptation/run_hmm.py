from __future__ import annotations

import argparse
import sys
from json import dump as jdump
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.append(str(Path(__file__).resolve().parent.parent))
import pickle

import pandas as pd

# from polygenic_adaptation.hmm_core import HMM
from hmm_core import HMM
from tqdm import tqdm


def compute_ll_wrapper(hmm, s, data_matrix, cond_endpt=False, **cond_kwargs):
    direc_res, direc_res_uncon = hmm.compute_multiple_ll(s / 2, s, data_matrix, cond_endpt, **cond_kwargs)
    stab_res, stab_res_uncon = hmm.compute_multiple_ll(s, 0, data_matrix, cond_endpt, **cond_kwargs)
    if np.any(np.isinf(stab_res)):
        pass
    return direc_res, direc_res_uncon, stab_res, stab_res_uncon


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

    parser.add_argument(
        "--subset_input",
        nargs=2,
        default=["", ""],
        help="use this flag if you are subsetting the full adna file on the fly",
    )
    parser.add_argument(
        "--condition_on_seg", action="store_true", help="condition the dataset on segregation at a future generation"
    )
    parser.add_argument(
        "--end_gen", type=int, help="generation to segregate based on (to be used with --condition_on_seg flag)"
    )
    parser.add_argument(
        "--end_ns", type=int, help="number of samples in final generation (to be used with --condition_on_seg flag)"
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

    if args_dict["condition_on_seg"]:
        assert "end_gen" in args_dict
        assert "end_ns" in args_dict
        args_dict["cond_kwargs"] = {"end_gen": args_dict["end_gen"], "end_ns": args_dict["end_ns"]}
    else:
        args_dict["cond_kwargs"] = {}

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

    if args_dict["subset_input"][0] != "" and data_matrix.shape[0] > 1:
        with Path(args_dict["subset_input"][0]).open("rb") as f:
            adna_snp_info = pickle.load(f)

        clumped_info = pd.read_parquet(args_dict["subset_input"][1])

        adna_mask = np.isin(adna_snp_info["all_rsid"], clumped_info["rsid"])
        data_matrix = data_matrix[adna_mask]

    MIN_GRID_VAL = 5e-5
    pos_grid = np.geomspace(MIN_GRID_VAL, args_dict["grid_s_max"], args_dict["num_half_grid_points"])
    full_grid = np.concatenate((-pos_grid[::-1], [0], pos_grid))

    np.linspace(-args_dict["grid_s_max"], args_dict["grid_s_max"], 1001)
    direc_unif_lls = np.zeros((data_matrix.shape[0], full_grid.shape[0]))
    direc_unif_uncon_lls = np.zeros((data_matrix.shape[0], full_grid.shape[0]))
    stab_unif_lls = np.zeros((data_matrix.shape[0], full_grid.shape[0]))
    stab_unif_uncon_lls = np.zeros((data_matrix.shape[0], full_grid.shape[0]))

    if args_dict["num_cores"] > 1:
        parallel_loop = tqdm(full_grid) if args_dict["progressbar"] else full_grid
        with Parallel(n_jobs=args_dict["num_cores"]) as parallel:
            res = parallel(
                delayed(compute_ll_wrapper)(
                    hmm, p_s, data_matrix, cond_endpt=args_dict["condition_on_seg"], **args_dict["cond_kwargs"]
                )
                for p_s in parallel_loop
            )
        direc_lls = [rp[0] for rp in res]
        direc_lls_uncon = [rp[1] for rp in res]
        stab_lls = [rp[2] for rp in res]
        stab_lls_uncon = [rp[3] for rp in res]
        direc_unif_lls[:, :] = np.array(direc_lls).T.squeeze()
        direc_unif_uncon_lls[:, :] = np.array(direc_lls_uncon).T.squeeze()
        stab_unif_lls[:, :] = np.array(stab_lls).T.squeeze()
        stab_unif_uncon_lls[:, :] = np.array(stab_lls_uncon).T.squeeze()
    else:
        for s_i, s in enumerate(tqdm(full_grid)) if args_dict["progressbar"] else enumerate(full_grid):
            # direc_unif_lls[:, s_i] = hmm.compute_multiple_ll(s1=s / 2, s2=s, data_matrix=data_matrix, cond_endpt=args_dict["condition_on_seg"], **args_dict["cond_kwargs"])
            # stab_unif_lls[:, s_i] = hmm.compute_multiple_ll(s1=s, s2=0, data_matrix=data_matrix, cond_endpt=args_dict["condition_on_seg"], **args_dict["cond_kwargs"])
            direc_unif_lls[:, s_i], direc_unif_uncon_lls[:, s_i], stab_unif_lls[:, s_i], stab_unif_uncon_lls[:, s_i] = (
                compute_ll_wrapper(hmm, s, data_matrix, args_dict["condition_on_seg"], **args_dict["cond_kwargs"])
            )

    combined_grid = np.zeros((2 * direc_unif_lls.shape[0] + 1, direc_unif_lls.shape[1]))
    combined_grid[0, :] = full_grid
    combined_grid[1::2, :] = direc_unif_lls
    combined_grid[2::2, :] = stab_unif_lls
    np.savetxt(
        args_dict["output_path"],
        combined_grid,
        header="s_grid followed by direc_unif_ll+stab_unif_ll for each rep",
    )
    if args_dict["condition_on_seg"]:
        combined_uncon_grid = np.copy(combined_grid)
        combined_uncon_grid[1::2, :] = direc_unif_uncon_lls
        combined_uncon_grid[2::2, :] = stab_unif_uncon_lls
        np.savetxt(
            f"{Path(args_dict['output_path']).with_suffix('')}_uncon.csv",
            combined_uncon_grid,
            header="s_grid followed by UNCON direc_unif_ll+stab_unif_ll for each rep",
        )
    if "snakemake" not in args_dict or not args_dict["snakemake"]:
        json_fname = f"{Path(args_dict['output_path']).with_suffix('')}_params.json"
        with Path(json_fname).open("w", encoding="locale") as file:
            jdump(args_dict, file)


if __name__ == "__main__":
    main()
