"""
"""

import argparse
import logging
import time
import sys

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

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()

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
