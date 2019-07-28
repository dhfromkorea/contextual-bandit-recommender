"""
Runner for fully observable reward CB problems.
"""
import os

import numpy as np
import pandas as pd


from datautils.synthetic.sample_data import sample_synthetic
from datautils.mushroom.sample_data import sample_mushroom
from datautils.preprocessing import load_data

from models.context_free_policy import (
        EpsilonGreedyPolicy,
        UCBPolicy,
)
from models.disjoint_contextual_policy import (
        LinUCBPolicy,
        LinearGaussianThompsonSamplingPolicy,
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_dir = os.path.abspath(os.path.join(root_dir, "results"))



def simulate_cb(data, n_samples, policies):
    """Simulator for FOR CB problems.

    Runs n_samples steps.
    """
    # infer T
    results = [None] * len(policies)

    for i, policy in enumerate(policies):
        results[i] = {}
        # log a_t, r_t, del_t (regret)
        results[i]["log"] = np.zeros((4, n_samples))

        t = 0

        for x_t, r_acts, a_t_opt, _ in zip(*data):
            a_t = policy.choose_action(x_t)
            r_t = r_acts[a_t]
            policy.update(a_t, x_t, r_t)
            r_t_opt = r_acts[a_t_opt]
            regret_t = r_t_opt - r_t

            results[i]["log"][:, t] = [a_t, a_t_opt, r_t, regret_t]

            t += 1

        results[i]["policy"] = policy
        regrets = results[i]["log"][3, :]
        results[i]["cum_regret"] = np.cumsum(regrets)
        results[i]["simple_regret"] = np.sum(regrets[-500:])

    return results


def run_cb(args):
    """Run fully observable reward CB problems.
    """
    task = args.task
    n_rounds = args.n_rounds

    if task == "mushroom":
        X, y = load_data(name="mushroom")
        context_dim = 117
        n_actions = 2

        samples = sample_mushroom(X,
                                  y,
                                  n_rounds,
                                  r_eat_good=10.0,
                                  r_eat_bad_lucky=10.0,
                                  r_eat_bad_unlucky=-50.0,
                                  r_eat_bad_lucky_prob=0.7,
                                  r_no_eat=0.0
                                  )

    elif task == "synthetic":
        n_actions = 5
        context_dim = 10
        sigma = 1.0 # set low covariance
        samples = sample_synthetic(n_rounds, n_actions, context_dim, sigma)

    else:
        raise NotImplementedError

    # define a solver
    egp = EpsilonGreedyPolicy(n_actions, lr=0.001,
                    epsilon=0.5, eps_anneal_factor=0.001)
    ucbp = UCBPolicy(n_actions=n_actions, lr=0.001)
    linucbp = LinUCBPolicy(
            n_actions=n_actions,
            context_dim=context_dim,
            delta=0.001,
            train_starts_at=100,
            train_freq=5
            )
    lgtsp = LinearGaussianThompsonSamplingPolicy(
                n_actions=n_actions,
                context_dim=context_dim,
                eta_prior=6.0,
                lambda_prior=0.25,
                train_starts_at=100,
                posterior_update_freq=5,
                lr = 0.05)

    policies = [egp, ucbp, linucbp, lgtsp]
    policy_names = ["egp", "ucbp", "linucbp", "lgtsp"]

    # simulate a bandit over n_rounds steps
    results = simulate_cb(samples, n_rounds, policies)

    return results, policies, policy_names


def write_results_cb(results, policies, policy_names, trial_idx, args):
    """Writes results to csv files.
    """
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

    acts_opt = results[0]["log"][1, :][:, None]
    acts_data = np.hstack( (acts_data, acts_opt) )

    df = pd.DataFrame(cumreg_data, columns=policy_names)
    df.to_csv("{}/{}.cumreg.{}.csv".format(results_dir, args.task, trial_idx), header=True, index=False)

    df = pd.DataFrame(acts_data, columns=policy_names + ["opt_p"])
    df.to_csv("{}/{}.acts.{}.csv".format(results_dir, args.task, trial_idx), header=True, index=False)
