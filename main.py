"""
"""
import time
import sys
import logging
from subprocess import Popen
import argparse

import torch

from runner_cb import run_context_bandit, write_results_cb
from runner_acb import run_action_context_bandit, write_results_acb


logger = logging.getLogger(__name__)
logging.basicConfig(
        filename="train.log",
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG
)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def arg_parser():
    parser = argparse.ArgumentParser()

    TASK_LIST = ["mushroom", "synthetic", "news"]

    parser.add_argument("task", type=str, choices=TASK_LIST)
    parser.add_argument("--n_trials", type=int, default=1, help="number of \
            independent trials for experiments")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_rounds", type=int, default=1000)
    parser.add_argument("--is_acp", action="store_true", help="whether the \
            task is an action context problem")

    # neural network stuff
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    #parser.add_argument('--n_rounds', type=int, default=1000, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.01')
    parser.add_argument('--grad_noise', action='store_true', help='add gradient noise')
    parser.add_argument('--eta', type=float, default=0.01, help='eta')
    parser.add_argument('--gamma', type=float, default=0.55, help='set gamme for guassian noise')
    parser.add_argument('--grad_clip', action='store_true', help='clip gradient')
    parser.add_argument('--grad_clip_norm', type=int, default=2, help='norm of the gradient clipping, default: l2')
    parser.add_argument('--grad_clip_value', type=float, default=10.0, help='the gradient clipping value')
    parser.add_argument('--n_worker', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--debug', action='store_true', help='enables debug mode')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for sgd')

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()

    for arg in vars(args):
        logging.info("{} - {}".format(arg, getattr(args, arg)) )

    if torch.cuda.is_available() and not args.cuda:
        logging.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

    logger.info("task: running {} trials with {} rounds".format(args.task, \
                args.n_trials, args.n_rounds))

    for trial_idx in range(args.n_trials):
        logger.info("{}th trial started".format(trial_idx))
        start_t = time.time()
        if args.is_acp:
            results, policies, policy_names = run_action_context_bandit(args)
            write_results_acb(results, policies, policy_names, trial_idx, args)
        else:
            results, policies, policy_names = run_context_bandit(args)
            write_results_cb(results, policies, policy_names, trial_idx, args)
        logger.info("{}th trial ended after {:.2f}s".format(trial_idx,
            time.time() - start_t))
