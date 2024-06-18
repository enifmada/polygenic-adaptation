import numpy as np
from numba import njit

def generate_poly_sims(*, num_gens: int = None, S_init: float = None, dz_init: float = None, Ne: float = None, betas: np.ndarray = None, p_init: np.ndarray = None, ziti: bool = False, rng_object, ) -> np.ndarray:
    N = p_init.shape[0]
    assert N == betas.shape[0]
    assert p_init.ndim == 1
    assert betas.ndim == 1

    if ziti:
        pass
    else:
        direc_coeffs = betas*S_init*dz_init
        stab_coeffs = betas**2 * S_init/2
        if not np.all(np.isclose(direc_coeffs, 0)):
            stab_coeffs = np.zeros_like(betas)
        full_res = np.zeros((N, num_gens+1))
        full_res[:, 0] = p_init
        for i in np.arange(num_gens):
            p_prime = full_res[:, i] + direc_coeffs * full_res[:, i] * (1-full_res[:, i]) - stab_coeffs * full_res[:, i] * (1-full_res[:, i]) * (1-2 * full_res[:, i])
            full_res[:, i+1] = rng_object.binomial((2*Ne), p_prime)/ (2*Ne)
            if np.all(full_res[:, i+1] <= 0):
                return full_res
            elif np.all(full_res[:, i+1] >= 1):
                full_res[:, i+2:] = 1
                return full_res
        return full_res

def generate_initial_condition(ic, num_loci, rng):
    if ic == "uniform":
        return rng.uniform(.05, .95, num_loci)
    else:
        print("not implemented yet!")
        pass
def generate_sampling_matrix(data_matched_tf: bool, num_loci, g_or_means, spt_or_missingness, nst_or_sampletable):
    if data_matched_tf:
        assert isinstance(g_or_means, str)
        assert isinstance(spt_or_missingness, str)
        assert isinstance(nst_or_sampletable, str)
        print("not implemented yet!")
        pass
    else:
        assert isinstance(g_or_means, int)
        assert isinstance(spt_or_missingness, int)
        assert isinstance(nst_or_sampletable, int)
        sampling_matrix = np.zeros((num_loci, g_or_means+1), dtype=int)
        sample_locs = np.linspace(0, sampling_matrix.shape[1]-1, nst_or_sampletable, dtype=int)
        sampling_matrix[:, sample_locs] = spt_or_missingness
        return sampling_matrix

def generate_betas(betas, num_loci, rng):
    if isinstance(betas, float):
        return np.zeros(num_loci, dtype=float) + betas
    elif isinstance(betas, str):
        return rng.choice(np.loadtxt(betas), size=num_loci, replace=True)
    else:
        raise TypeError("Invalid type of --beta_coefficients!")

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
        chebyshev_pts = 1/2 + np.cos((2*np.arange(1,n_total-1)-1)*np.pi/(2*(n_total-2)))/2
        all_pts = np.concatenate((np.array([0]), chebyshev_pts[::-1], np.array([1])))
        return all_pts, generate_bounds(all_pts)
    if hidden_interp == "linear":
        return np.linspace(0,1,n_total), generate_bounds(np.linspace(0,1,n_total))
    raise TypeError("Invalid hidden interpolation function!")

def generate_bounds(states):
    bounds = np.zeros(states.shape[0]+1)
    bounds[1:-1] = states[:-1] + np.diff(states)/2
    bounds[0] = states[0]-(states[1]-states[0])/2
    bounds[-1] = states[-1] + (states[-1]-states[-2])/2
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
    return fname[:-1]