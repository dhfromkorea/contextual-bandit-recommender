"""
Plotting util.
Assumes experiment csv fils in the root.
"""

import os
import argparse


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = os.path.abspath(os.path.dirname(__file__))


def arg_parser():
    TASK_LIST = ["mushroom", "synthetic", "news"]

    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, choices=TASK_LIST)
    parser.add_argument("--n_trials", type=int, default=1, help="number of \
            independent trials for experiments")

    parser.add_argument("--window", type=int,
                        default=100, help="moving average window")
    return parser.parse_args()


def main():
    args = arg_parser()
    sns.set()
    sns.set_palette("husl")

    if args.task in ["news"]:
        plot_acb(args.task, args)

    if args.task in ["mushroom", "synthetic"]:
        plot_cb(args.task, args)


def compute_mean_std(paths, n_trials, window=-1):
    M_cr = []
    for i in range(n_trials):
        df_cr = pd.read_csv(paths[i])
        # numpy
        if window == -1:
            M_cr.append(df_cr.to_numpy())
        else:
            # smooth data
            smoothed_data = df_cr.rolling(window, 1, axis=0).mean()
            M_cr.append(smoothed_data.to_numpy())

    M_cr_mean = None
    for i in range(n_trials):
        if M_cr_mean is None:
            M_cr_mean = M_cr[i]
        else:
            M_cr_mean += M_cr[i]
    M_cr_mean = M_cr_mean / n_trials

    M_cr_std = np.zeros( M_cr_mean.shape )
    M_cr_std_low = np.zeros( M_cr_mean.shape )
    for i in range(n_trials):
        M_cr_std = np.maximum(M_cr[i], M_cr_std)
        M_cr_std_low = np.minimum(M_cr[i], M_cr_std_low)

    return M_cr_mean, (M_cr_std, M_cr_std_low), df_cr.columns


def plot_acb(task, args):
    n_trials = args.n_trials
    window = args.window
    paths = []
    for i in range(n_trials):
        p = os.path.join(root_dir, "{}.cumrew.{}.csv".format(task, i))
        paths.append(p)
    M_cr_mean, M_cr_std, columns = compute_mean_std(paths, n_trials)
    plot_cumrew(task, M_cr_mean, M_cr_std, columns)

    paths = []
    for i in range(n_trials):
        p = os.path.join(root_dir, "{}.CTR.{}.csv".format(task, i))
        paths.append(p)
    M_cr_mean, M_cr_std, columns = compute_mean_std(paths, n_trials,
            window=window)

    plot_CTR(task, M_cr_mean, M_cr_std, columns, window)


def plot_cb(task, args):
    n_trials = args.n_trials
    paths = []
    for i in range(n_trials):
        p = os.path.join(root_dir, "{}.cumreg.{}.csv".format(task, i))
        paths.append(p)
    M_cr_mean, M_cr_std, columns = compute_mean_std(paths, n_trials)
    plot_cumreg(task, M_cr_mean, M_cr_std, columns)

    paths = []
    for i in range(n_trials):
        p = os.path.join(root_dir, "{}.acts.{}.csv".format(task, i))
        paths.append(p)
    # @todo: fix this
    paths = [paths[0]]
    plot_acts(task, paths, columns)


def plot_cumreg(task_name, M_cr_mean, M_cr_std, columns):
    high, low = M_cr_std
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title("Cumulative Regret ({})".format(task_name), fontsize=25)
    ax.set_xlabel("Rounds (t)", fontsize=25)
    ax.set_ylabel("Cumulative Regret", fontsize=25)

    x = np.arange(M_cr_mean.shape[0])
    for j in range(M_cr_mean.shape[1]):
        y_mean = M_cr_mean[:, j]
        ax.plot(x, y_mean + j * 0.01, label=columns[j], linewidth=1)
        ax.fill_between(x, np.maximum(0, low[:,j]), high[:,j], alpha=0.1)

    ax.legend(loc="upper left", fontsize=15)
    fig.savefig("{}.cumreg.png".format(task_name), bbox_inches="tight")
    plt.close()


def plot_acts(task_name, paths, columns):
    plt.figure(figsize=(7, 7))
    plt.title("Action Log ({})".format(task_name), fontsize=25)
    plt.xlabel("Rounds (t)", fontsize=25)
    plt.ylabel("Chosen action (a_t)", fontsize=25)

    for p in paths:
        # for each trial data
        df = pd.read_csv(p)
        columns = df.columns
        M_acts = df.to_numpy()
        for j in range(M_acts.shape[1]):
            y = M_acts[:, j]
            plt.scatter(np.arange(M_acts.shape[0]), y + j * 0.05,
                    label=columns[j], s=1)
        plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, markerscale=6,
                fontsize=15)
    plt.savefig("{}.acts.png".format(task_name), bbox_inches="tight")
    plt.close()


def plot_cumrew(task_name, M_cr_mean, M_cr_std, columns):
    high, low = M_cr_std
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title("Cumulative Reward ({})".format(task_name), fontsize=25)
    ax.set_xlabel("Rounds (t)", fontsize=25)
    ax.set_ylabel("Cumulative Reward", fontsize=25)

    x = np.arange(M_cr_mean.shape[0])
    for j in range(M_cr_mean.shape[1]):
        y_mean = M_cr_mean[:, j]
        ax.plot(x, y_mean, label=columns[j], linewidth=1)
        ax.fill_between(x, np.maximum(0, low[:, j]), high[:, j], alpha=0.1)

    ax.legend(loc="upper left", fontsize=15)
    fig.savefig("{}.cumrew.png".format(task_name), bbox_inches="tight")
    plt.close()


def plot_CTR(task_name, M_cr_mean, M_cr_std, columns, window):
    high, low = M_cr_std
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_ylim([0.0, 0.4])
    ax.set_title("CTR (smoothing={}) ({})".format(window, task_name), fontsize=25)
    ax.set_xlabel("Rounds (t)", fontsize=25)
    ax.set_ylabel("CTR", fontsize=25)

    x = np.arange(M_cr_mean.shape[0])
    for j in range(M_cr_mean.shape[1]):
        y_mean = M_cr_mean[:, j]
        ax.plot(x, y_mean, label=columns[j])
        ax.fill_between(x, np.maximum(0, low[:,j]), high[:,j], alpha=0.1)

    ax.legend(loc="upper left", fontsize=15)
    fig.savefig("{}.CTR.png".format(task_name), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
