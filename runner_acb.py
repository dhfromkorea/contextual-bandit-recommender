
import numpy as np
import pandas as pd
import torch


from datasets.news.sample_data import sample_user_event
from datasets.bandit_data import BanditData

from models.context_free_policy import (
        RandomPolicy
)
from models.action_context_based_policy import (
        SharedLinUCBPolicy,
        SharedLinearGaussianThompsonSamplingPolicy,
        FeedForwardNetwork,
        NeuralPolicy,
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
            train_starts_at=args.train_starts_at,
            train_freq=args.train_freq
    )

    lgtsp = SharedLinearGaussianThompsonSamplingPolicy(
                context_dim=context_dim,
                eta_prior=6.0,
                lambda_prior=0.25,
                train_starts_at=args.train_starts_at,
                posterior_update_freq=args.train_freq
    )

    # prepare nueral policy

    np.random.seed(0)
    torch.manual_seed(0)

    batch_size = args.batch_size
    set_gpu = args.cuda
    eta = args.eta
    gamma = args.gamma

    grad_clip = args.grad_clip
    grad_clip_norm = args.grad_clip_norm
    grad_clip_value = args.grad_clip_value

    grad_noise = args.grad_noise


    ffn = FeedForwardNetwork(input_dim=context_dim,
                              hidden_dim=64,
                              output_dim=1,
                              n_layer=3,
                              learning_rate=args.lr,
                              set_gpu=set_gpu,
                              grad_noise=grad_noise,
                              gamma=gamma,
                              eta=eta,
                              grad_clip=grad_clip,
                              grad_clip_norm=grad_clip_norm,
                              grad_clip_value=grad_clip_value,
                              weight_decay=args.weight_decay,
                              debug=args.debug)

    # batch data loader
    bd = BanditData(batch_size, epoch_len=16)
    # 16 x 64

    neuralp = NeuralPolicy(ffn, bd, train_starts_at=args.train_starts_at,
            train_freq=args.train_freq, set_gpu=set_gpu)

    policies = [rp, linucbp, lgtsp, neuralp]
    policy_names = ["rp", "linucbp", "lgtsp", "neuralp"]
    #policies = [neuralp]
    #policy_names = ["neuralp"]

    import time
    start_t = time.time()
    results = simulate_contextual_bandit_partial_label(uv_generator, n_rounds, policies)
    print("took {}s / 1 trial with {}".format(time.time() - start_t,
        policy_names[0]))

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


