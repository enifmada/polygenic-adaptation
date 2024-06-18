import numpy as np
from scipy.stats import norm, binom, beta
from polyutil import get_uq_a_exps, generate_states_new

class HMM:
    def __init__(self, num_approx: int, Ne: float, init_cond: str = "theta", **kwargs):
        assert num_approx > 0
        self.init_cond = init_cond
        self.custom_init_cond = False
        self.N = num_approx
        self.Ne = Ne

        self.gs, self.bounds = generate_states_new(self.N,"chebyshev")
        self.qs = 1-self.gs
        self.gs_product = np.multiply.outer(self.gs, self.gs)

        self.integral_bounds = self.bounds.copy()
        self.integral_bounds[0] = -np.inf
        self.integral_bounds[-1] = np.inf

        #self.a_ij = P(S_t+1 = j|S_t = i)
        if self.init_cond == "uniform":
            pre_istate = np.diff(self.bounds)
            self.init_state = pre_istate/np.sum(pre_istate)
        elif self.init_cond == "delta":
            self.init_state = np.zeros_like(self.gs)
            self.init_state[np.clip(np.argmin(np.abs(self.gs-kwargs["p"])), 1, self.N-2)] = 1.
        elif self.init_cond == "beta":
            self.init_state = np.zeros_like(self.gs)
            beta_param = kwargs["beta_coef"]
            beta_distrib = beta(beta_param, beta_param)
            beta_pdf = beta_distrib.pdf(self.gs[1:-1])
            self.init_state[1:-1] = beta_pdf/np.sum(beta_pdf)
        elif self.init_cond == "spikeandslab":
            self.init_state = np.ones_like(self.gs)
            self.init_state /= np.sum(self.init_state)
            self.init_state *= 1-kwargs["spike_frac"]
            self.init_state[np.clip(np.argmin(np.abs(self.gs - kwargs["spike_loc"])), 1, self.N - 2)] += kwargs["spike_frac"]
        else:
            raise TypeError("Invalid initial condition specification!")

        assert np.isclose(np.sum(self.init_state), 1.)
        self.init_init_state = np.copy(self.init_state)

    def calc_transition_probs_old(self, s_vector):
        s1 = s_vector[0]
        s2 = s_vector[1]
        p_primes = np.clip(self.gs + self.gs * self.qs * (s2 * self.gs + s1 * (1 - 2 * self.gs)), 0, 1)
        sigmas = np.sqrt(self.gs * self.qs / (2*self.Ne))
        a_one = np.zeros((1, self.gs.shape[0]))
        a_one[:, 0] = 1.
        a_all = np.concatenate((a_one, np.diff(norm.cdf(np.expand_dims(self.integral_bounds, axis=-1),
                                            p_primes[1:-1], sigmas[1:-1]), axis=0).T, a_one[:, ::-1]), axis=0)
        return a_all

    def clip_and_renormalize(self, matrix, val):
        uniform_matrix = np.diff(self.bounds)
        uniform_matrix /= np.sum(uniform_matrix)
        if uniform_matrix[0] < val:
            raise ValueError
        scale_factor = val/uniform_matrix[0]
        int_matrix = (1-scale_factor)*(matrix/np.sum(matrix)) + scale_factor*uniform_matrix
        int_matrix /= np.sum(int_matrix)
        return int_matrix

    def compute_multiple_ll(self, s1, s2, data_matrix):
        sample_locs_array = data_matrix[:, ::3]
        nts_array = data_matrix[:, 1::3]
        obs_counts_array = data_matrix[:, 2::3]
        ll_T = int(sample_locs_array[0,-1] + 1)
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
        ll_bpmf_a = np.zeros(np.sum(ll_nts_uq) + ll_nts_uq.shape[0])
        ll_bpmf_n = np.zeros_like(ll_bpmf_a)

        ll_bpmf_idx = np.cumsum(ll_nts_uq + 1)
        for i, nt in enumerate(ll_nts_uq[1:], 1):
            ll_bpmf_a[ll_bpmf_idx[i - 1]:ll_bpmf_idx[i]] = np.arange(nt + 1)
            ll_bpmf_n[ll_bpmf_idx[i - 1]:ll_bpmf_idx[i]] = nt

        ll_b = binom
        # self.bpmf = self.b.pmf(np.broadcast_to(self.obs_counts[..., None], self.obs_counts.shape+(self.N,)), np.broadcast_to(self.nts[..., None], self.nts.shape+(self.N,)), np.broadcast_to(self.gs, (self.nloc, self.T, self.N))).transpose([1,2,0])
        ll_bpmf_new = ll_b.pmf(np.broadcast_to(ll_bpmf_a[..., None], ll_bpmf_a.shape + (self.N,)),
                                   np.broadcast_to(ll_bpmf_n[..., None], ll_bpmf_n.shape + (self.N,)),
                                   np.broadcast_to(self.gs, (ll_bpmf_a.shape[0], self.N)))

        ll_a_t_to_bpmf_idx = np.zeros_like(ll_nts)
        for i, t in np.transpose(np.nonzero(ll_nts)):
            ll_a_t_to_bpmf_idx[i, t] = ll_obs_counts[i, t] + ll_bpmf_idx[
                np.where(ll_nts_uq == ll_nts[i, t])[0][0] - 1]

        ll_a = self.calc_transition_probs_old([s1, s2])
        assert np.all(np.isclose(np.sum(ll_a, axis=1), 1))
        sample_times = np.nonzero(np.any(ll_nts, axis=0))[0]
        sample_time_diffs = np.diff(sample_times)
        uq_a_powers = np.unique(sample_time_diffs)
        uq_a_exps = get_uq_a_exps(ll_a, uq_a_powers)
        ll_cs = np.ones((ll_T, ll_nloc))
        ll_alphas_tilde = np.einsum("i, ni->in", self.init_state, ll_bpmf_new[ll_a_t_to_bpmf_idx[:, 0], :])
        ll_cs[0, :] = 1. / np.sum(ll_alphas_tilde, axis=0)
        ll_alphas_hat = np.einsum("n, in -> in", ll_cs[0, :], ll_alphas_tilde)
        for i, t in enumerate(sample_times[1:]):
            ll_alphas_tilde = np.einsum("in, ij, nj -> jn", ll_alphas_hat,
                    uq_a_exps[np.where(uq_a_powers == sample_time_diffs[i])[0][0]],ll_bpmf_new[ll_a_t_to_bpmf_idx[:, t], :])
            ll_cs[t, :] = 1. / np.sum(ll_alphas_tilde, axis=0)
            ll_alphas_hat = np.einsum("n, in -> in", ll_cs[t, :], ll_alphas_tilde)
            assert np.all(np.isclose(np.sum(ll_alphas_hat, axis=0), 1.))
        return -np.sum(np.log(ll_cs), axis=0)