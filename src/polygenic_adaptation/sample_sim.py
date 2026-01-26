from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np


def get_params(fname, vars, prefixes):
    vars_dict = {}
    flags_dict = {}
    for var in vars:
        vars_dict[var] = 0
        flags_dict[var] = False
    fparts = fname.split("_")
    for fpart in fparts[1:]:
        if fpart[0] in prefixes and not flags_dict[vars[prefixes.index(fpart[0])]]:
            vars_dict[vars[prefixes.index(fpart[0])]] = float(fpart[1:])
            flags_dict[vars[prefixes.index(fpart[0])]] = True
        # this is awful
        elif fpart[:2] in prefixes and not flags_dict[vars[prefixes.index(fpart[:2])]]:
            vars_dict[vars[prefixes.index(fpart[:2])]] = float(fpart[2:])
            flags_dict[vars[prefixes.index(fpart[:2])]] = True
    return vars_dict


def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", nargs="*", help="input")
    parser.add_argument("-o", "--output", nargs="*", help="output")
    parser.add_argument(
        "--sampling_scheme",
        type=str,
        required=True,
        help="mode of sampling to use; currently either 'fixed' or 'matched'",
    )
    parser.add_argument(
        "--samples_per_timepoint", type=int, help="number of samples per timepoint (for use with 'fixed')"
    )
    parser.add_argument("--num_sampling_pts", type=int, help="number of sampling timepoints (for used with 'fixed')")
    parser.add_argument(
        "--sampling_table_file",
        type=str,
        help="txt file containing the number of samples at each timepoint to use (for use with 'matched')",
    )
    parser.add_argument("--betas_file", type=str, help="path to effect sizes at each site. used for plotting.")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--sim_source", type=str, help="slim vs polysim")
    smk = parser.parse_args()

    assert smk.sampling_scheme in ["fixed", "matched"], "sampling method not yet implemented!"
    assert smk.sim_source in ["polysim", "slim"]

    known_vars = ["omega", "seed", "loci", "h2"]
    known_prefixes = ["w", "s", "l", "h"]

    rng = np.random.default_rng(smk.seed)
    vars_dict = get_params(smk.input[0], known_vars, known_prefixes)
    if smk.sim_source == "slim":
        slim_array = np.loadtxt(smk.input[0], skiprows=1).T
        freqs_sub_const = 3
    else:
        slim_array = np.load(smk.input[0])["arr_0"]
        freqs_sub_const = 0

    if smk.sampling_scheme == "fixed":
        final_array = np.zeros((slim_array.shape[0] - freqs_sub_const, smk.num_sampling_pts * 3))
        sampling_times = np.linspace(0, slim_array.shape[1] - 1, smk.num_sampling_pts, dtype=int)
        final_array[:, ::3] = sampling_times
        final_array[:, 1::3] = smk.samples_per_timepoint
        final_array[:, 2::3] = rng.binomial(smk.samples_per_timepoint, slim_array[freqs_sub_const:, sampling_times])
    elif smk.sampling_scheme == "matched":
        sampling_table = np.loadtxt(smk.sampling_table_file, skiprows=1, dtype=int)
        sampling_table[:, 0] -= np.min(sampling_table[:, 0])
        final_array = np.zeros((slim_array.shape[0] - freqs_sub_const, sampling_table.shape[0] * 3))
        final_array[:, ::3] = sampling_table[:, 0]
        final_array[:, 1::3] = sampling_table[:, 1]
        final_array[:, 2::3] = rng.binomial(sampling_table[:, 1], slim_array[freqs_sub_const:, sampling_table[:, 0]])

    betas = np.loadtxt(smk.betas_file)

    pop_genvar = np.sum(
        2 * betas[:, np.newaxis] ** 2 * slim_array[freqs_sub_const:, :] * (1 - slim_array[freqs_sub_const:, :]),
        axis=0,
    )
    p_samp = final_array[:, 2::3] / final_array[:, 1::3]
    samp_genvar = np.sum(2 * betas[:, np.newaxis] ** 2 * p_samp * (1 - p_samp), axis=0)
    np.savetxt(
        smk.output[0],
        final_array,
        delimiter="\t",
        fmt="%d",
        header="Each row = one replicate; each set of three columns = (sampling time; total samples; derived alleles)",
    )
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
    axs[0].plot(slim_array[freqs_sub_const:, :].T)
    axs[1].plot(final_array[:, ::3].T, final_array[:, 2::3].T)
    axs[2].plot(np.arange(slim_array.shape[1]), pop_genvar, label="pop")
    axs[2].plot(final_array[0, ::3], samp_genvar, label="samp")
    axs[2].legend()
    axs[2].set_title(rf"$\omega^2=${vars_dict['omega']**2}")
    fig.savefig(smk.output[1], format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
