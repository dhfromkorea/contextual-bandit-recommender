"""
"""
import numpy as np
import pandas as pd
from pprint import pprint as pp


from datasets.synthetic.sample_data import sample_synthetic
from datasets.mushroom.sample_data import sample_mushroom
from datasets.preprocessing import load_data

from models.context_free_policy import EpsilonGreedyPolicy, RandomPolicy, SampleMeanPolicy, UCBPolicy
from models.context_based_policy import LinUCBPolicy, LinearGaussianThompsonSamplingPolicy
from simulate import simulate_contextual_bandit

# we should forget about small efficiencies
# say about 97% of the time
# premature optimization is the root of the evil
# yet, we should not pass up opportunities
# for the critical 3%


def main(args):
    """TODO: Docstring for main.

    Parameters
    ----------
    arg1 : TODO

    Returns
    -------
    TODO

    """

    task = args["task"]

    if task == "mushroom":
        X, y = load_data(name="mushroom")
        # simulate the problem T steps
        n_samples = 5 * (10 ** 4)
        #n_samples = 10000
        context_dim = 117
        n_actions = 2

        # optimal strategy
        # if good -> eat
        # if bad -> not eat

        samples = sample_mushroom(X,
                                  y,
                                  n_samples,
                                  r_eat_good=5.0,
                                  r_eat_bad_lucky=5.0,
                                  r_eat_bad_unlucky=-35.0,
                                  r_eat_bad_lucky_prob=0.5,
                                  r_no_eat=0.0
                                  )

    elif task == "synthetic":
        n_samples = 5000
        n_actions = 5
        context_dim = 10
        sigma = 1.0 # set low covariance
        samples = sample_synthetic(n_samples, n_actions, context_dim, sigma)

    else:
        raise NotImplementedError


    # define a solver
    rp = RandomPolicy(n_actions)
    smp = SampleMeanPolicy(n_actions)
    egp = EpsilonGreedyPolicy(n_actions, lr=0.1, epsilon=0.1)
    ucbp = UCBPolicy(n_actions=n_actions, lr=0.01)
    linucbp = LinUCBPolicy(
            n_actions=n_actions,
            context_dim=context_dim,
            delta=0.25,
            train_starts_at=500,
            train_freq=50)
    lgtsp = LinearGaussianThompsonSamplingPolicy(
                n_actions=n_actions,
                context_dim=context_dim,
                eta_prior=6.0,
                lambda_prior=0.25,
                train_starts_at=500,
                posterior_update_freq=50
            )

    policies = [rp, smp, egp, ucbp, linucbp, lgtsp]
    policy_names = ["rp", "smp", "egp", "ucbp", "linucbp", "lgtsp"]


    # simulate a bandit over n_samples steps
    results = simulate_contextual_bandit(samples, n_samples, policies)


    # log results
    cumreg_data = None
    acts_data = None
    for i in range(len(policies)):
        cr = results[i]["cum_regret"][:, None]
        if cumreg_data is None:
            cumreg_data = cr
        else:
            cumreg_data = np.hstack( (cumreg_data, cr) )

        acts = results[i]["log"][0, :][:, None]
        if acts_data is None:
            acts_data = acts
        else:
            acts_data = np.hstack( (acts_data, acts) )

    # add opt_acts
    # assume it's the same across policies
    acts_opt = results[0]["log"][1, :][:, None]
    acts_data = np.hstack( (acts_data, acts_opt) )


    df = pd.DataFrame(cumreg_data, columns=policy_names)
    df.to_csv("{}.cumreg.csv".format(task), header=True, index=False)

    df = pd.DataFrame(acts_data, columns=policy_names + ["opt_p"])
    df.to_csv("{}.acts.csv".format(task), header=True, index=False)


if __name__ == "__main__":
    args = {}
    #args["task"] = "mushroom"
    #main(args)
    args["task"] = "synthetic"
    main(args)

