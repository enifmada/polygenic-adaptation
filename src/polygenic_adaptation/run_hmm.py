from hmm_core import HMM
import numpy as np
from json import dump as jdump
from tqdm import tqdm
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="path to input dataset")
    parser.add_argument("output_path", type=str, help="path to output csv - additional files may be created in the same directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--time_before_present", action="store_true", help="dates provided start at a number at the earliest time and count down towards the present")
    group.add_argument("--time_after_zero", action="store_true", help="dates provided start at zero at the earliest time and count up towards the present")
    parser.add_argument("--grid_s_max", type=float, default=.25, help="maximum s value to use in the grid")
    parser.add_argument("-np", "--num_half_grid_points", type=int, default=15, help="number of half-grid points to use - the full grid will have 2*np + 1 points")
    parser.add_argument("-ytg", "--years_to_gen", type=float, default=1, help="years per generation in VCF or CSV")
    parser.add_argument("-hs", "--hidden_states", type=int, help="number of approx states in HMM", default=500)
    parser.add_argument("-sid", "--starting_init_dist", default="uniform", help="initial initial condition to use")
    parser.add_argument("--sid_dict", nargs='*', default="", help="initial condition dictionary")
    parser.add_argument("-Ne", type=int, default=10000, help="effective population size for the HMM")
    parser.add_argument("--progressbar", action="store_true", help="adds a tqdm progress bar")
    parser.add_argument("--save_csv", action="store_true", help="if inputting a VCF, save a CSV to future reduce pre-processing time")
    parser.add_argument("--info_file", type=argparse.FileType("rb"), help="sample times file (if input = VCF)")
    parser.add_argument("--info_cols", type=str, nargs=2, default=["Genetic_ID","Date_mean"], help="names of the ID and dates columns in the sample times file (if input = VCF)")
    parser.add_argument("--full_output", action="store_true", help="save a pickle file with a full set of outputs (in addition to the CSV)")
    parser.add_argument("--force", type=str, nargs=1, help="if the VCF file only contains homozygous loci, force it to be read as either haploid or diploid")
    parser.add_argument("--snakemake", action="store_true",
                        help="whether or not this script was run as part of a snakemake workflow. If so, do not save the params as a json because params.json already exists.")
    args_dict = vars(parser.parse_args())
    actual_sid_dict = {}

    if args_dict["sid_dict"] is not None:
        for ic_pair in args_dict["sid_dict"]:
            k, v = ic_pair.split('=')
            try:
                actual_sid_dict[k] = float(v)
            except:
                actual_sid_dict[k] = v
    args_dict["sid_dict"] = actual_sid_dict
    hmm = HMM(args_dict["hidden_states"], args_dict["Ne"], args_dict["starting_init_dist"], **args_dict["sid_dict"])

    if Path(args_dict["input_path"]).suffix == ".csv":
        data_matrix = np.loadtxt(args_dict["input_path"], dtype=int)
        print(data_matrix.shape[0])
    else:
        pass
    MIN_GRID_VAL = 1e-8
    pos_grid = np.geomspace(MIN_GRID_VAL, args_dict["grid_s_max"], args_dict["num_half_grid_points"])
    full_grid = np.concatenate((-pos_grid[::-1], [0], pos_grid))
    direc_lls = np.zeros((data_matrix.shape[0], full_grid.shape[0]))
    stab_lls = np.zeros((data_matrix.shape[0], full_grid.shape[0]))

    enum_object = enumerate(tqdm(full_grid)) if args_dict["progressbar"] else enumerate(full_grid)
    for s_i, s in enum_object:
        direc_lls[:, s_i] = hmm.compute_multiple_ll(s1=s/2, s2=s, data_matrix=data_matrix)
        stab_lls[:, s_i] = hmm.compute_multiple_ll(s1=s, s2=0, data_matrix=data_matrix)

    combined_grid = np.zeros((2*direc_lls.shape[0]+1, direc_lls.shape[1]))
    combined_grid[0, :] = full_grid
    for row in range(direc_lls.shape[0]):
        combined_grid[2*row+1, :] = direc_lls[row, :]
        combined_grid[2*row+2, :] = stab_lls[row, :]
    print(combined_grid.shape)
    np.savetxt(args_dict["output_path"],combined_grid,header="s_grid followed by direc_ll, stab_ll for each rep")
    if "snakemake" not in args_dict.keys() or not args_dict["snakemake"]:
        json_fname = f"{Path(args_dict['output_path']).with_suffix('')}_params.json"
        with open(json_fname, "w") as file:
            jdump(args_dict, file)


if __name__ == "__main__":
    main()