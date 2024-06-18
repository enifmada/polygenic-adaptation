from polyutil import generate_poly_sims, generate_initial_condition, generate_sampling_matrix, generate_betas, generate_fname
import numpy as np
import matplotlib.pyplot as plt
import argparse
from itertools import product as itprod
from json import dump as jdump
def float_or_str(val):
    try:
        return float(val)
    except:
        return val
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=str, help="path to output directory")
    parser.add_argument("-n", "--num_reps", type=int, default=16, help="number of replicates - NOT number of loci per sim")
    parser.add_argument("-nl", "--num_loci", type=int, default=64, help="number of loci per replicate")
    parser.add_argument("-S", "--sel_gradients", nargs="+", type=float, default=1, help="selection coefficients to simulate")
    parser.add_argument("-dz", "--delta_z", nargs="+", type=float, default=.1, help="starting distance to optimum of fitness surface")
    parser.add_argument("-g", "--num_gens", nargs="+", type=int, default=300, help="number of generations to simulate")
    parser.add_argument("-ic", "--init_conds", nargs="+", type=float_or_str, default="uniform", help="initial conditions to simulate")
    parser.add_argument("-b", "--beta_coefficients", type=float_or_str, default=.1, help="value of the effect size for each locus OR path to a txt file to sample effect sizes from")
    parser.add_argument("--data_matched", type=str, nargs=3, default=["", "", ""],
                        help="input the path to means + missingness .txt files + sampling .table, will override -g, -ic, -nss and -spt to initialize and sample according to the table")
    parser.add_argument("-spt", "--samples_per_timepoint", type=int, default=30, help="number of samples to draw at each sampling timepoint")
    parser.add_argument("-nst", "--num_sampling_times", type=int, default=11, help="number of times to draw samples")
    parser.add_argument("-Ne", type=int, default=10000, help="effective population size")
    parser.add_argument("--seed", type=int, default=6, help="seed")
    parser.add_argument("--save_plots", action="store_true", help="save plots of all of the replicate trajectories")
    parser.add_argument("--prefix", type=str, default="", help="file names prefix to differentiate runs")
    parser.add_argument("--snakemake", action="store_true", help="whether or not this script was run as part of a snakemake workflow. If so, do not save the params as a json because params.json already exists.")
    args_dict = vars(parser.parse_args())
    for S, dz, g, ic in itprod(args_dict["sel_gradients"], args_dict["delta_z"], args_dict["num_gens"], args_dict["init_conds"]):
        ns_matrix = generate_sampling_matrix(args_dict["data_matched"][0] != "", args_dict["num_loci"],
                                             g, args_dict["samples_per_timepoint"],
                                             args_dict["num_sampling_times"])
        base_fname = generate_fname(**{"S":S, "dz":dz, "g":g, "ic":ic})
        print(base_fname)
        for i in range(args_dict["num_reps"]):
            rng = np.random.default_rng(args_dict["seed"]+i)
            p_init = generate_initial_condition(ic, args_dict["num_loci"], rng)
            betas = generate_betas(args_dict["beta_coefficients"], args_dict["num_loci"], rng)
            add_gen_var = np.sum(2 * betas ** 2 * p_init * (1 - p_init))
            print(f"additive genetic variance: {add_gen_var:.4f}")
            freqs = generate_poly_sims(num_gens=g, S_init = S, dz_init = dz, Ne = args_dict["Ne"], betas = betas, p_init = p_init, ziti=False, rng_object =rng)

            assert freqs.shape == ns_matrix.shape

            #sample freqs
            nonzero_sample_locs = np.any(ns_matrix > 0, axis=0)
            sampled_freqs = rng.binomial(ns_matrix[:, nonzero_sample_locs], freqs[:, nonzero_sample_locs])

            data_matrix = np.zeros((freqs.shape[0], sampled_freqs.shape[1]*3))
            data_matrix[:, ::3] = np.where(nonzero_sample_locs)[0]
            data_matrix[:, 1::3] = ns_matrix[:, nonzero_sample_locs]
            data_matrix[:, 2::3] = sampled_freqs

            fig, axs = plt.subplots(1,1,figsize=(10,10),layout="constrained")
            axs.plot(freqs.T)
            fig.savefig(f"{args_dict['output_directory']}/{args_dict['prefix']}{base_fname}_rep{i}_freqs.pdf", format="pdf", bbox_inches="tight")
            plt.close(fig)

            np.savetxt(f"{args_dict['output_directory']}/{args_dict['prefix']}{base_fname}_rep{i}_data.csv", data_matrix, delimiter="\t", fmt="%d",
                header="Each row = one replicate; each set of three columns = (sampling time; total samples; derived alleles)")

    if "snakemake" not in args_dict.keys() or not args_dict["snakemake"]:
        json_fname = f"{args_dict['output_directory']}/{args_dict['prefix']}_params.json"
        with open(json_fname, "w") as file:
            jdump(args_dict, file)

if __name__ == "__main__":
    main()
