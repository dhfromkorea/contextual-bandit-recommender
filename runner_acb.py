import numpy as np
import pandas as pd


from datasets.news.sample_data import sample_user_event

from models.context_free_policy import (
        RandomPolicy
)
from models.action_context_based_policy import (
        SharedLinUCBPolicy,
        SharedLinearGaussianThompsonSamplingPolicy,
)
from simulate import (
        simulate_contextual_bandit_partial_label,
)


def run_action_context_bandit(args):
    """
    runner for problems that have action context

    action context = features for each valid action
    """

    n_rounds = args.n_rounds
    uv_generator = sample_user_event()


    n_actions = 20
    context_dim = 6 + 6

    rp = RandomPolicy(n_actions)

    linucbp = SharedLinUCBPolicy(
            context_dim=context_dim,
            delta=0.25,
            train_starts_at=300,
            train_freq=500
    )

    lgtsp = SharedLinearGaussianThompsonSamplingPolicy(
                context_dim=context_dim,
                eta_prior=6.0,
                lambda_prior=0.25,
                train_starts_at=300,
                posterior_update_freq=500
    )


    policies = [rp, linucbp, lgtsp]
    policy_names = ["rp", "linucbp", "lgtsp"]

    results = simulate_contextual_bandit_partial_label(uv_generator, n_rounds, policies)

    return results, policies, policy_names


def write_results_acb(results, policies, policy_names, trial_idx, args):
    """
    write results to csv for action context problems.
    """
    # log results
    cumrew_data = None
    for i in range(len(policies)):
        cr = results[i]["cum_reward"][:, None]
        if cumrew_data is None:
            cumrew_data = cr
        else:
            cumrew_data = np.hstack( (cumrew_data, cr) )


    df = pd.DataFrame(cumrew_data, columns=policy_names)
    df.to_csv("{}.cumrew.{}.csv".format(args.task, trial_idx), header=True, index=False)


    CTR_data = None
    for i in range(len(policies)):
        cr = results[i]["CTR"][:, None]
        if CTR_data is None:
            CTR_data = cr
        else:
            CTR_data = np.hstack( (CTR_data, cr) )


    df = pd.DataFrame(CTR_data, columns=policy_names)
    df.to_csv("{}.CTR.{}.csv".format(args.task, trial_idx), header=True, index=False)

