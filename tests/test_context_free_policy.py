import pytest


import numpy as np


from context import EpsilonGreedyPolicy, UCBPolicy
from context import simulate_contextual_bandit
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
def test_epsilon_greedy_policy(n_samples, n_actions, context_dim, dataset):
    # define a solver
    egp = EpsilonGreedyPolicy(n_actions=n_actions, lr=0.1, epsilon=0.1)
    policies = [egp]

    results = simulate_contextual_bandit(dataset, n_samples, policies)

    # no operational error
    assert results[0]["simple_regret"] > -1.0


@pytest.mark.parametrize(
        "n_samples, n_actions, context_dim, dataset",
        [synthetic_data, mushroom_data]
)
def test_ucb_policy(n_samples, n_actions, context_dim, dataset):
    # define a solver
    ucbp = UCBPolicy(n_actions=n_actions, lr=0.01)
    policies = [ucbp]

    results = simulate_contextual_bandit(dataset, n_samples, policies)

    # no operational error
    assert results[0]["simple_regret"] > -1.0


