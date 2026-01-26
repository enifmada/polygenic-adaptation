from __future__ import annotations

import contextlib
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("-l", "--num_loci", help="number of loci to simulate")
    parser.add_argument("-n", type=int, help="number of individuals to simulate")
    parser.add_argument("-g", "--num_gens", type=int, help="number of generations to simulate")
    parser.add_argument("--vs", type=float, help="value of v_s to simulate under")
    parser.add_argument("-dz", "--delta_z", type=float, help="value of delta z to simulate under")
    parser.add_argument("-h2", type=float, help="value of h2 to simulate under")
    parser.add_argument("--sel_type", help="type of selection to simulate. currently directional or stabilizing")
    parser.add_argument("--betas_file", type=str, help="path to betas file")
    parser.add_argument("--freqs_file", type=str, help="path to initial frequencies file")
    parser.add_argument("-o", "--output", nargs="*", help="output, in freqs, phenos, genos order")
    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument(
        "--big_gwas",
        action="store_true",
        help="artificially increase the population size in the last generation for the GWAS",
    )

    smk = parser.parse_args()

    betas_array = np.loadtxt(smk.betas_file)
    freqs_array = np.loadtxt(smk.freqs_file)

    assert betas_array.shape[0] == freqs_array.shape[0]

    if smk.num_loci == "all":
        num_loci = betas_array.shape[0]
    else:
        with contextlib.suppress(TypeError, ValueError):
            num_loci = int(smk.num_loci)

    assert smk.sel_type in ("stabilizing", "directional")
    assert np.all(freqs_array >= 0)
    assert np.all(freqs_array <= 1)
    assert smk.h2 >= 0
    assert smk.h2 <= 1
    assert smk.vs >= 0
    assert num_loci > 0
    assert smk.n > 0

    vg_init = 2 * np.sum(betas_array**2 * freqs_array * (1 - freqs_array), axis=0)
    ve = (vg_init - smk.h2 * vg_init) / smk.h2

    rng = np.random.default_rng(smk.seed)

    p_t = np.zeros((num_loci, smk.num_gens + 1))
    p_t[:, 0] = freqs_array
    init_states = np.zeros((num_loci, smk.n * 2), dtype=bool)
    for l_i in np.arange(num_loci):
        init_states[l_i, : int(p_t[l_i, 0] * 2 * smk.n)] = 1
        rng.shuffle(init_states[l_i])
    g_t = init_states

    if smk.sel_type == "stabilizing":
        haplo_phenos = np.sum(g_t * betas_array[:, np.newaxis], axis=0)
        ind_phenos = haplo_phenos[::2] + haplo_phenos[1::2]
        z0 = np.mean(ind_phenos)

    for gen_i in tqdm(np.arange(smk.num_gens)):
        z_haplo = np.sum(g_t * betas_array[:, np.newaxis], axis=0)
        z_ind = z_haplo[::2] + z_haplo[1::2]
        if ve > 0:
            z_ind += rng.normal(0, scale=np.sqrt(ve), size=z_ind.shape[0])
        # compute w(z)
        if smk.sel_type == "stabilizing":
            w_t = np.exp(-((z_ind - z0) ** 2) / (2 * smk.vs))
        else:
            w_t = np.exp((z_ind * smk.delta_z) / (smk.vs))

        # reproduce
        g_tp1 = np.zeros_like(g_t)
        big_haplo_rng = rng.uniform(size=(num_loci, smk.n * 2))
        parents = rng.choice(np.arange(smk.n), size=(smk.n * 2), p=w_t / np.sum(w_t))
        for ind_i in np.arange(smk.n * 2):
            g_tp1[:, ind_i] = np.where(
                big_haplo_rng[:, ind_i] < 0.5, g_t[:, parents[ind_i] * 2], g_t[:, parents[ind_i] * 2 + 1]
            )
        p_t[:, gen_i + 1] = g_tp1.sum(axis=-1) / (2 * smk.n)
        g_t = g_tp1
    if smk.big_gwas:
        biggg_haplo_rng = rng.uniform(size=(num_loci, smk.n * 10))
        big_parents = rng.choice(np.arange(smk.n), size=(smk.n * 10), p=w_t / np.sum(w_t))
        g_t_final = np.zeros((num_loci, smk.n * 10), dtype=bool)
        for ind_i in np.arange(smk.n * 10):
            g_t_final[:, ind_i] = np.where(
                biggg_haplo_rng[:, ind_i] < 0.5, g_t[:, big_parents[ind_i] * 2], g_t[:, big_parents[ind_i] * 2 + 1]
            )
    else:
        g_t_final = g_t
    z_haplo_final = np.sum(g_t_final * betas_array[:, np.newaxis], axis=0)
    z_ind_final = z_haplo_final[::2] + z_haplo_final[1::2]
    if ve > 0:
        z_ind_final += rng.normal(0, scale=np.sqrt(ve), size=z_ind_final.shape[0])

    # output
    np.savez_compressed(smk.output[0], p_t)
    np.savez_compressed(smk.output[1], z_ind_final)
    np.savez_compressed(smk.output[2], g_t_final)


if __name__ == "__main__":
    main()
