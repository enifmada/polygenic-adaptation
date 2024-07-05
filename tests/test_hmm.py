from __future__ import annotations

import numpy as np
import pytest

from polygenic_adaptation.hmm_core import HMM


@pytest.fixture()
def example_sampled_data():
    return np.array([[0,5,5,25,5,5,50,5,5,75,5,5,100,5,5],[0,5,0,25,5,1,50,5,2,75,5,3,100,5,4]])

def test_neutral_ll(example_sampled_data):
    test_hmm = HMM(num_approx=500, Ne=10000,init_cond="uniform")
    lls = test_hmm.compute_multiple_ll(0,0,example_sampled_data)
    np.testing.assert_allclose(np.array([-3.23774647, -10.38231924]), lls)

def test_sel_lls(example_sampled_data):
    test_hmm = HMM(num_approx=500, Ne=10000, init_cond="uniform")
    lls = test_hmm.compute_multiple_ll(.1, .2, example_sampled_data)
    np.testing.assert_allclose(np.array([-1.86616165, -12.79082467]), lls)