import pytest
import numpy as np

from context import sample_synthetic

@pytest.fixture(scope="module")
def synthetic_data():
    n_samples = 1000
    n_actions = 5
    context_dim = 10
    samples = sample_synthetic(n_samples, n_actions, context_dim)
    return n_samples, n_actions, samples


def test_sample_synthetic(synthetic_data):
    n_samples, n_actions, samples = synthetic_data

    x_t, r_acts, opt_act, mean_hidden = next(samples)

    assert len(r_acts) == n_actions
    assert opt_act == np.argmax(mean_hidden)



