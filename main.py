"""
"""
import numpy as np
from pprint import pprint as pp


from datasets.mushroom.sample_data import sample_mushroom
from datasets.preprocessing import load_data
from models.context_free_policy import EpsilonGreedyPolicy, RandomPolicy, SampleMeanPolicy, UCBPolicy
from models.context_based_policy import LinUCBPolicy, LinUCBHybridPolicy, LinearRegressorPolicy
from simulate import simulate_contextual_bandit

# we should forget about small efficiencies
# say about 97% of the time
# premature optimization is the root of the evil
# yet, we should not pass up opportunities
# for the critical 3%


def main(arg1):
    """TODO: Docstring for main.

    Parameters
    ----------
    arg1 : TODO

    Returns
    -------
    TODO

    """

    X, y = load_data(name="mushroom")


    # define a solver
    rp = RandomPolicy(n_actions=2)
    smp = SampleMeanPolicy(n_actions=2, lr=0.1)
    egp = EpsilonGreedyPolicy(n_actions=2, lr=0.1, epsilon=0.1)
    ucbp = UCBPolicy(n_actions=2, lr=0.01)
    lrp = LinearRegressorPolicy(n_actions=2)
    linucbp = LinUCBPolicy(n_actions=2)

    policies = [rp, smp, egp, ucbp, linucbp]


    # simulate the problem T steps
    T = 5 * (10 ** 4)
    mushrooms = sample_mushroom(X,
                                y,
                                T,
                                r_eat_good=5.0,
                                r_eat_bad_lucky=5.0,
                                r_eat_bad_unlucky=-35.0,
                                r_eat_bad_lucky_prob=0.5,
                                r_no_eat=0.0
                                )

    results = simulate_contextual_bandit(mushrooms, policies)

    pp(results)


# thompson sampling
class ThompsonSampling(object):
    """Docstring for ThompsonSampling. """

    def __init__(self):
        """TODO: to be defined1. """
        pass

if __name__ == "__main__":
    args = []
    main(args)

