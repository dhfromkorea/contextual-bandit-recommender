"""
"""
import numpy as np
from pprint import pprint as pp


from datasets.mushroom.sample_data import sample_mushroom
from datasets.preprocessing import load_data
from methods.context_free_policy import EpsilonGreedyPolicy, RandomPolicy, SampleMeanPolicy
from methods.context_based_policy import LinUCBPolicy, LinUCBHybridPolicy, LinearRegressorPolicy

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

    results = {}


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

    for i, policy in enumerate(policies):

        rewards = np.zeros(T)
        regrets = np.zeros(T)
        t = 0

        # can regret be negative?
        for c_t, r_eat_t, r_no_eat_t, a_opt in zip(*mushrooms):
            a_t = policy.choose_action(c_t)
            r_t = a_t * r_eat_t + (1 - a_t) * r_no_eat_t

            policy.update(a_t, c_t, r_t)

            r_t_opt =  a_opt * r_eat_t + (1 - a_opt) * r_no_eat_t

            rewards[t] = r_t
            regrets[t] = r_t_opt - r_t

            t += 1

        results[i] = {
                #"regrets": regrets,
                #"rewards": rewards,
                "cum_regrets": np.sum(regrets),
                "simple_regret": np.mean(regrets[-500:])
                }

        #results[i] = (np.mean(rewards), np.mean(regrets))

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

