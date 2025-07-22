from __future__ import annotations

import numpy as np
from numba import njit


def generate_poly_sims(
    *,
    num_gens: int | None = None,
    S_init: float | None = None,
    dz_init: float | None = None,
    Ne: float | None = None,
    betas: np.ndarray = None,
    p_init: np.ndarray = None,
    ziti: bool = False,
    rng_object,
) -> np.ndarray:
    N = p_init.shape[0]
    assert betas.shape[0] == N
    assert p_init.ndim == 1
    assert betas.ndim == 1

    if ziti:
        return None
    direc_coeffs = betas * S_init * dz_init
    stab_coeffs = betas**2 * S_init / 2
    if not np.all(np.isclose(direc_coeffs, 0)):
        stab_coeffs = np.zeros_like(betas)
    full_res = np.zeros((N, num_gens + 1))
    full_res[:, 0] = p_init
    for i in np.arange(num_gens):
        p_prime = (
            full_res[:, i]
            + direc_coeffs * full_res[:, i] * (1 - full_res[:, i])
            - stab_coeffs * full_res[:, i] * (1 - full_res[:, i]) * (1 - 2 * full_res[:, i])
        )
        full_res[:, i + 1] = rng_object.binomial((2 * Ne), p_prime) / (2 * Ne)
        if np.all(full_res[:, i + 1] <= 0):
            return full_res
        if np.all(full_res[:, i + 1] >= 1):
            full_res[:, i + 2 :] = 1
            return full_res
    return full_res


def generate_initial_condition(ic, num_loci, rng):
    if ic == "uniform":
        return rng.uniform(0.05, 0.95, num_loci)
    return None


def generate_sampling_matrix(data_matched_tf: bool, num_loci, g_or_means, spt_or_missingness, nst_or_sampletable):
    if data_matched_tf:
        assert isinstance(g_or_means, str)
        assert isinstance(spt_or_missingness, str)
        assert isinstance(nst_or_sampletable, str)
        return None
    assert isinstance(g_or_means, int)
    assert isinstance(spt_or_missingness, int)
    assert isinstance(nst_or_sampletable, int)
    sampling_matrix = np.zeros((num_loci, g_or_means + 1), dtype=int)
    sample_locs = np.linspace(0, sampling_matrix.shape[1] - 1, nst_or_sampletable, dtype=int)
    sampling_matrix[:, sample_locs] = spt_or_missingness
    return sampling_matrix


def generate_betas(betas, num_loci, rng, **kwargs):
    if isinstance(betas, float):
        return np.zeros(num_loci, dtype=float) + betas
    if isinstance(betas, str):
        if betas == "uniform":
            if "std_frac_err" in kwargs:
                betas = rng.normal(0, kwargs["std"], size=num_loci)
                betas_hat = rng.normal(betas, np.abs(betas * kwargs["std_frac_err"]), size=num_loci)
                return betas, betas_hat
            return rng.normal(0, kwargs["std"], size=num_loci)
        return rng.choice(np.loadtxt(betas), size=num_loci, replace=True)
    msg = "Invalid type of --beta_coefficients!"
    raise TypeError(msg)


@njit
def get_uq_a_exps(a, powers):
    a_exps_list = [np.linalg.matrix_power(a, powers[0])]
    powers_diff = np.diff(powers)
    for power_diff in powers_diff:
        if power_diff == 1:
            a_exps_list.append(a_exps_list[-1] @ a)
        else:
            a_exps_list.append(a_exps_list[-1] @ np.linalg.matrix_power(a, power_diff))
    return a_exps_list


def generate_states_new(n_total, hidden_interp):
    if hidden_interp == "chebyshev":
        chebyshev_pts = 1 / 2 + np.cos((2 * np.arange(1, n_total - 1) - 1) * np.pi / (2 * (n_total - 2))) / 2
        all_pts = np.concatenate((np.array([0]), chebyshev_pts[::-1], np.array([1])))
        return all_pts, generate_bounds(all_pts)
    if hidden_interp == "linear":
        return np.linspace(0, 1, n_total), generate_bounds(np.linspace(0, 1, n_total))
    msg = "Invalid hidden interpolation function!"
    raise TypeError(msg)


def generate_bounds(states):
    bounds = np.zeros(states.shape[0] + 1)
    bounds[1:-1] = states[:-1] + np.diff(states) / 2
    bounds[0] = states[0] - (states[1] - states[0]) / 2
    bounds[-1] = states[-1] + (states[-1] - states[-2]) / 2
    return bounds


def generate_fname(**kwargs):
    fname = ""
    fkeys = kwargs.keys()
    if "g" in fkeys:
        fname += f"g{kwargs['g']}_"
    if "S" in fkeys:
        fname += f"S{kwargs['S']}_"
    if "dz" in fkeys:
        fname += f"dz{kwargs['dz']}_"
    if "ic" in fkeys:
        fname += f"{kwargs['ic']}_"
    if "std_err" in fkeys:
        fname += f"err{kwargs['std_err']:.2f}_"
    return fname[:-1]


def vcf_to_useful_format(vcf_file, sample_times_file, years_per_gen=28.1, force=None):
    sample_times_ordered = np.copy(sample_times_file)
    sample_times_ordered[:, 1] //= years_per_gen
    max_sample_time = np.max(sample_times_ordered[:, 1])
    sample_times_ordered = sample_times_ordered[np.argsort(sample_times_ordered[:, 0]), :]
    correct_order_idxs = vcf_file["samples"].argsort().argsort()
    sample_times_ordered = sample_times_ordered[correct_order_idxs, :]
    sample_times, sample_idxs = np.unique(sample_times_ordered[:, 1], return_inverse=True)
    chroms = vcf_file["variants/CHROM"].astype(int)

    # if we're doing genome-wide thresholds?
    np.any(vcf_file["calldata/GT"][:, :, 1] >= 0, axis=0)
    if np.all(vcf_file["calldata/GT"][:, :, 0] == vcf_file["calldata/GT"][:, :, 1]):
        if not force or force not in ["haploid", "diploid"]:
            msg = "VCF call data is all homozygotes - must use --force [haploid/diploid]!"
            raise TypeError(msg)
        if force == "haploid":
            vcf_file["calldata/GT"][:, :, 1] = -1
    big_final_table = np.zeros((1, sample_times.shape[0] * 3))
    for chrom in np.unique(chroms):
        final_table = np.zeros(((chroms == chrom).sum(), sample_times.shape[0] * 3))
        for sample_i in range(sample_times.shape[0]):
            sample_indices = np.where(sample_i == sample_idxs)[0]
            assert (vcf_file["samples"][sample_indices] == sample_times_ordered[sample_indices, 0]).all()
            relevant_calls_nd = np.squeeze(vcf_file["calldata/GT"][:, sample_indices, :])
            num_samples_nd = np.sum(relevant_calls_nd >= 0, axis=-1)
            num_zeros_nd = np.sum(relevant_calls_nd == 0, axis=-1)
            if relevant_calls_nd.ndim > 2:
                num_samples_nd = np.sum(num_samples_nd, axis=1)
                num_zeros_nd = np.sum(num_zeros_nd, axis=1)
            final_data_nd = num_samples_nd - num_zeros_nd
            final_table[:, sample_i * 3 + 1] = num_samples_nd.astype(int)
            final_table[:, sample_i * 3 + 2] = final_data_nd.astype(int)
        final_table[:, ::3] = (max_sample_time - sample_times[::-1]).astype(int)
        final_table[:, 1::3] = final_table[:, 1::3][:, ::-1]
        final_table[:, 2::3] = final_table[:, 2::3][:, ::-1]
        big_final_table = np.vstack((big_final_table, final_table))
    return big_final_table[1:, :]


def bh_correct(p_values, alpha, yekutieli=False):
    M = p_values.shape[0]
    p_values_sorted = np.sort(p_values.copy())
    bh_range = np.arange(1, M + 1)
    if yekutieli:
        alpha /= np.sum(1 / bh_range)
    small_ps = np.where(p_values_sorted <= bh_range * alpha / M)[0]
    if small_ps.shape[0] > 0:
        k_max = np.where(p_values_sorted <= bh_range * alpha / M)[0][-1]
    else:
        return 1, np.array([])
    p_k = np.sqrt(p_values_sorted[k_max] * p_values_sorted[k_max + 1])
    return p_k, np.where(p_values <= p_k)[0]
