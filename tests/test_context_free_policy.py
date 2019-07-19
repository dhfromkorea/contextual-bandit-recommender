import pytest


import numpy as np


from context import *


@pytest.fixture
def mushrooms():
    X, y = load_data(name="mushroom")

    T = 100000
    # sample the problem T steps for simulation

    # scenario - optimal strategy
    # E[R_t | eat good] = 5.0
    # E[R_t | eat bad] = 0.9 * 5.0 + 0.1 * -5.0 = 4.0
    # 4.0 <= E[R_t | eat] <= 5.0
    # E[R_t | no_eat] = 2.0

    # hence, should always eat
    mushrooms = sample_mushroom(X,
                                y,
                                T,
                                r_eat_good=5.0,
                                r_eat_bad_lucky=5.0,
                                r_eat_bad_unlucky=-5.0,
                                r_eat_bad_lucky_prob=0.9,
                                r_no_eat=2.0
                                )
    return mushrooms

def test_epsilon_greedy_policy(mushrooms):

    # define a solver
    egp = EpsilonGreedyPolicy(n_actions=2, lr=0.1, epsilon=0.1)
    policies = [egp]

    results = simulate_contextual_bandit(mushrooms, policies)

    # must avoid getting stuck at no eating
    assert results[0]["simple_rewards"] > 2.0




def test_ucb_policy(mushrooms):

    # define a solver
    ucbp = UCBPolicy(n_actions=2, lr=0.01)
    policies = [ucbp]


    results = simulate_contextual_bandit(mushrooms, policies)

    # must avoid getting stuck at no eating
    assert results[0]["simple_rewards"] > 2.0

