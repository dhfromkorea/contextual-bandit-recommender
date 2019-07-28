import pytest


import numpy as np


from context import LinUCBPolicy
from context import LinearGaussianThompsonSamplingPolicy
from context import simulate_cb
from context import sample_synthetic, sample_mushroom


def get_mushroom_data():
    context_dim = 5
    n_samples = 5000
    n_actions = 2

    # mock mushroom data
    X = np.random.rand(n_samples, context_dim)
    # relate y to X
    get_norm = lambda x_i : np.linalg.norm(x_i) > 2.5
    y = np.apply_along_axis(get_norm, 1, X)
    #y = np.random.choice([0, 1], size=n_samples)

    # sample the problem T steps for simulation

    # should always eat when good
    # avoid eating when bad
    samples = sample_mushroom(X,
                              y,
                              n_samples,
                              r_eat_good=5.0,
                              r_eat_bad_lucky=5.0,
                              r_eat_bad_unlucky=-35.0,
                              r_eat_bad_lucky_prob=0.1,
                              r_no_eat=0.0
                              )
    return n_samples, n_actions, context_dim, samples


def get_synthetic_data():
    n_samples = 5000
    n_actions = 3
    context_dim = 5
    sigma = 1.0
    samples = sample_synthetic(n_samples, n_actions, context_dim, sigma)
    return n_samples, n_actions, context_dim, samples


# manual fixture
mushroom_data = get_mushroom_data()
synthetic_data = get_synthetic_data()


@pytest.mark.parametrize(
        "n_samples, n_actions, context_dim, dataset",
        [synthetic_data, mushroom_data]
)
def test_linucb_policy(n_samples, n_actions, context_dim, dataset):
    # define a solver
    linucbp = LinUCBPolicy(
            n_actions=n_actions,
            context_dim=context_dim,
            delta=0.25,
            train_starts_at=500,
            train_freq=50)

    policies = [linucbp]

    results = simulate_cb(dataset, n_samples, policies)

    # must avoid getting stuck at no eating
    # not sure about synthetic
    assert results[0]["simple_regret"] > -1.0


@pytest.mark.parametrize(
        "n_samples, n_actions, context_dim, dataset",
        [synthetic_data, mushroom_data]
)
def test_linear_gaussian_thompson_sampling_policy(
        n_samples, n_actions, context_dim, dataset):

    lgtsp = LinearGaussianThompsonSamplingPolicy(
                n_actions=n_actions,
                context_dim=context_dim,
                eta_prior=6.0,
                lambda_prior=0.25,
                train_starts_at=500,
                posterior_update_freq=50
            )


    policies = [lgtsp]

    results = simulate_cb(dataset, n_samples, policies)

    # must avoid getting stuck at no eating
    # not sure about synthetic
    assert results[0]["simple_regret"] > -1.0

