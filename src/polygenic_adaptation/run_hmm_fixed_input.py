from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import binom

sys.path.append(str(Path(__file__).resolve().parent.parent))
from tqdm import tqdm

from polygenic_adaptation.hmm_core import HMM
from polygenic_adaptation.polyutil import get_uq_a_exps


def compute_ll_wrapper(hmm, s, data_matrix, init_states):
    direc_res = hmm.compute_multiple_ll(s / 2, s, data_matrix, init_states)
    stab_res = hmm.compute_multiple_ll(s, 0, data_matrix, init_states)
    return direc_res, stab_res

#@jit
def compute_multiple_ll_numba_outer(a_matrix, data_matrix, N, gs, hmm_init_state):
    sample_locs_array = data_matrix[:, ::3]
    nts_array = data_matrix[:, 1::3]
    obs_counts_array = data_matrix[:, 2::3]
    ll_T = int(sample_locs_array[0, -1] + 1)
    ll_nloc = obs_counts_array.shape[0]
    ll_sample_locs = sample_locs_array[0, :]

    ll_obs_counts = np.zeros((ll_nloc, ll_T))
    ll_obs_counts[:, ll_sample_locs] = obs_counts_array
    ll_nts = np.zeros((ll_nloc, ll_T), dtype=int)
    ll_nts[:, ll_sample_locs] = nts_array

    # self.b.pmf(count[t], nts[t], gs[i]) = P(a_t|n_t, f_ti)!
    # self.bpmf[t, i, n] = P(a_n_t|n_n_t, f_n_ti)

    # self.bpmf_new[k/n, i] = P(a_t = k|n_t = n, f_i)
    ll_nts_uq = np.unique(ll_nts)

    # emission probability mass function (bpmf)
    # self.bpmf_new = np.zeros((np.sum(self.nts_uq)+self.nts_uq.shape[0], self.N))
    #??????
    ll_bpmf_a = np.zeros(np.sum(ll_nts_uq) + ll_nts_uq.shape[0])
    ll_bpmf_n = np.zeros_like(ll_bpmf_a)

    ll_bpmf_idx = np.cumsum(ll_nts_uq + 1)
    for i, nt in enumerate(ll_nts_uq[1:], 1):
        ll_bpmf_a[ll_bpmf_idx[i - 1] : ll_bpmf_idx[i]] = np.arange(nt + 1)
        ll_bpmf_n[ll_bpmf_idx[i - 1] : ll_bpmf_idx[i]] = nt

    ll_b = binom
    # self.bpmf = self.b.pmf(np.broadcast_to(self.obs_counts[..., None], self.obs_counts.shape+(self.N,)), np.broadcast_to(self.nts[..., None], self.nts.shape+(self.N,)), np.broadcast_to(self.gs, (self.nloc, self.T, self.N))).transpose([1,2,0])
    ll_bpmf_new = ll_b.pmf(
        np.broadcast_to(ll_bpmf_a[..., None], (*ll_bpmf_a.shape, N)),
        np.broadcast_to(ll_bpmf_n[..., None], (*ll_bpmf_n.shape, N)),
        np.broadcast_to(gs, (ll_bpmf_a.shape[0], N)),
    )

    ll_a_t_to_bpmf_idx = np.zeros_like(ll_nts)
    for i, t in np.transpose(np.nonzero(ll_nts)):
        ll_a_t_to_bpmf_idx[i, t] = (
            ll_obs_counts[i, t]
            + ll_bpmf_idx[np.where(ll_nts_uq == ll_nts[i, t])[0][0] - 1]
        )
    sample_times = np.nonzero(np.any(ll_nts, axis=0))[0]
    ll_alphas_tilde = np.zeros((hmm_init_state.shape[0], ll_a_t_to_bpmf_idx.shape[0]))
    ll_alphas_hat = np.zeros_like(ll_alphas_tilde)
    return compute_multiple_ll_numba_inner(a_matrix, sample_times, ll_alphas_tilde, ll_alphas_hat, ll_T, ll_nloc, ll_bpmf_new, ll_a_t_to_bpmf_idx, hmm_init_state)

def compute_multiple_ll_numba_inner(a_matrix, sample_times, ll_alphas_tilde, ll_alphas_hat, ll_T, ll_nloc, ll_bpmf_new, ll_a_t_to_bpmf_idx, hmm_init_state):
    sample_time_diffs = np.diff(sample_times)
    uq_a_powers = np.unique(sample_time_diffs)
    uq_a_exps = get_uq_a_exps(a_matrix, uq_a_powers)
    ll_cs = np.ones((ll_T, ll_nloc))
    init_obs = ll_bpmf_new[ll_a_t_to_bpmf_idx[:, 0], :]
    for i in range(hmm_init_state.shape[0]):
        ll_alphas_tilde[i, :] = hmm_init_state[i] * init_obs[:, i]

    # @staticmethod
    # @njit(fastmath=True)
    # def forward_one_numba(init_state, trans_matrix, a_t_to_bpmf, bpmf):
    #     N = init_state.shape[0]
    #     T = a_t_to_bpmf.shape[0]
    #     alphas_hat = np.zeros((T, N))
    #     cs = np.ones(T)
    #     cs[0] = 1.0 / np.sum(init_state * bpmf[a_t_to_bpmf[0], :])
    #     alphas_hat[0, :] = cs[0] * init_state * bpmf[a_t_to_bpmf[0], :]
    #     for t in np.arange(1, T):
    #         alphas_tilde = bpmf[a_t_to_bpmf[t], :] * np.dot(
    #             alphas_hat[t - 1, :], trans_matrix
    #         )
    #         cs[t] = 1.0 / np.sum(alphas_tilde)
    #         alphas_hat[t, :] = cs[t] * alphas_tilde
    #     return alphas_hat.T, cs
    ll_cs[0, :] = 1.0 / np.sum(ll_alphas_tilde, axis=0)
    for n in range(ll_cs[0, :].shape[0]):
        ll_alphas_hat[:, n] = ll_cs[0, n] * ll_alphas_tilde[:, n]
    for i, t in enumerate(sample_times[1:]):
        temp_a_matrix = uq_a_exps[np.where(uq_a_powers == sample_time_diffs[i])[0][0]]
        for inner_j in range(temp_a_matrix.shape[1]):
            for inner_n in range(ll_alphas_hat.shape[1]):
                # temp_mat_val = 0
                # for inner_i in range(ll_alphas_hat.shape[0]):
                #     temp_mat_val += ll_alphas_hat[inner_i, inner_n] * temp_a_matrix[inner_i, inner_j] * ll_bpmf_new[ll_a_t_to_bpmf_idx[inner_n, t], inner_j]
                # ll_alphas_tilde[inner_j, inner_n] = temp_mat_val
                ll_alphas_tilde[inner_j, inner_n] = np.sum(ll_alphas_hat[:, inner_n] * temp_a_matrix[:, inner_j]) * ll_bpmf_new[ll_a_t_to_bpmf_idx[inner_n, t], inner_j]
        ll_cs[t, :] = 1.0 / np.sum(ll_alphas_tilde, axis=0)
        for n in range(ll_cs[t, :].shape[0]):
            for inner_jj in range(ll_alphas_hat.shape[0]):
                ll_alphas_hat[inner_jj, n] = ll_cs[t, n] * ll_alphas_tilde[inner_jj, n]
        assert np.all(np.isclose(np.sum(ll_alphas_hat, axis=0), 1.0))
    return -np.sum(np.log(ll_cs), axis=0)

def main():
    num_hidden_states = 500
    Ne = 10000
    init_dist = "uniform"
    input_path = "../../../polyoutput/slim_testing/gwasfull/data/gwasfull_b0.015_w0.2_s0_data.csv"
    output_path = "../../../polyoutput/slim_testing/gwasfull/grids/gwasfull_b0.015_w0.2_s0_data_profiling.csv"
    grid_s_max = 0.05
    num_grid_points = 21
    hmm = HMM(num_hidden_states, Ne,init_dist)

    data_matrix = np.loadtxt(input_path, dtype=int)


    MIN_GRID_VAL = 5e-5
    pos_grid = np.geomspace(MIN_GRID_VAL, grid_s_max, num_grid_points)
    full_grid = np.concatenate((-pos_grid[::-1], [0], pos_grid))

    direc_unif_lls = np.zeros((data_matrix.shape[0], full_grid.shape[0]))
    stab_unif_lls = np.zeros((data_matrix.shape[0], full_grid.shape[0]))

    parallel_loop = tqdm(full_grid)
    with Parallel(n_jobs=7) as parallel:
        res = parallel(
            delayed(compute_ll_wrapper)(hmm, p_s, data_matrix, None)
            for p_s in parallel_loop
        )
    direc_lls = [rp[0] for rp in res]
    stab_lls = [rp[1] for rp in res]
    direc_unif_lls[:, :] = np.array(direc_lls).T.squeeze()
    stab_unif_lls[:, :] = np.array(stab_lls).T.squeeze()
    # for s_i, s in enumerate(tqdm(full_grid)):
    #     direc_unif_lls[:, s_i] = hmm.compute_multiple_ll(s/2, s, data_matrix, None)
    #         #compute_multiple_ll_numba_outer(hmm.calc_transition_probs_old([s/2, s]), data_matrix, hmm.N, hmm.gs, hmm.init_state)
    #     stab_unif_lls[:, s_i] = hmm.compute_multiple_ll(s, 0, data_matrix, None)
    #         #compute_multiple_ll_numba_outer(hmm.calc_transition_probs_old([s, 0]), data_matrix, hmm.N, hmm.gs, hmm.init_state)

    combined_grid = np.zeros((2 * direc_unif_lls.shape[0] + 1, direc_unif_lls.shape[1]))
    combined_grid[0, :] = full_grid
    for row in range(direc_unif_lls.shape[0]):
        combined_grid[2 * row + 1, :] = direc_unif_lls[row, :]
        combined_grid[2 * row + 2, :] = stab_unif_lls[row, :]
    np.savetxt(
        output_path,
        combined_grid,
        header="s_grid followed by direc_unif_ll+stab_unif_ll for each rep",)

if __name__ == "__main__":
    main()