import os


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = os.path.abspath(os.path.dirname(__file__))

def main():
    sns.set()
    sns.set_palette("husl")

    tasks = ["mushroom", "synthetic"]
    for task in tasks:
        cumreg_path = os.path.join(root_dir, "{}.cumreg.csv".format(task))
        acts_path = os.path.join(root_dir, "{}.acts.csv".format(task))

        df_cumreg = pd.read_csv(cumreg_path)
        df_acts = pd.read_csv(acts_path)

        plot_cumreg(task, df_cumreg)
        plot_acts(task, df_acts)

    tasks = ["news"]
    for task in tasks:
        cumrew_path = os.path.join(root_dir, "{}.cumrew.csv".format(task))
        df_cumrew = pd.read_csv(cumrew_path)
        plot_cumrew(task, df_cumrew)

        CTR_path = os.path.join(root_dir, "{}.CTR.csv".format(task))
        df_CTR = pd.read_csv(CTR_path)
        plot_CTR(task, df_CTR)

def plot_cumreg(task_name, df_cumreg):
    plt.figure(figsize=(7, 7))
    plt.title("Cumulative Regret ({})".format(task_name), fontsize=25)
    plt.xlabel("Rounds (t)", fontsize=25)
    plt.ylabel("Cumulative Regret", fontsize=25)
    for j in range(df_cumreg.shape[1]):
        y = df_cumreg.iloc[:, j]
        plt.plot(np.arange(df_cumreg.shape[0]), y, label=df_cumreg.columns[j])
    #plt.legend(bbox_to_anchor=(1,0), loc="lower right",
    #          bbox_transform=fig.transFigure, ncol= df_cumreg.shape[1])
    plt.legend(loc="upper left", fontsize=15)
    plt.savefig("{}.cumreg.png".format(task_name), bbox_inches="tight")
    plt.close()


def plot_acts(task_name, df_acts):
    plt.figure(figsize=(7, 7))
    plt.title("Action Log ({})".format(task_name), fontsize=25)
    plt.xlabel("Rounds (t)", fontsize=25)
    plt.ylabel("Chosen action (a_t)", fontsize=25)
    for j in range(df_acts.shape[1]):
        y = df_acts.iloc[:, j]
        plt.scatter(np.arange(df_acts.shape[0]), y + j * 0.05,
                label=df_acts.columns[j], s=3)
    plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, markerscale=6,
            fontsize=15)
    plt.savefig("{}.acts.png".format(task_name), bbox_inches="tight")
    plt.close()


def plot_cumrew(task_name, df_cumrew):
    plt.figure(figsize=(7, 7))
    plt.title("Cumulative Reward ({})".format(task_name), fontsize=25)
    plt.xlabel("Rounds (t)", fontsize=25)
    plt.ylabel("Cumulative Reward", fontsize=25)
    for j in range(df_cumrew.shape[1]):
        y = df_cumrew.iloc[:, j]
        plt.plot(np.arange(df_cumrew.shape[0]), y, label=df_cumrew.columns[j])
    #plt.legend(bbox_to_anchor=(1,0), loc="lower right",
    #          bbox_transform=fig.transFigure, ncol= df_cumreg.shape[1])
    plt.legend(loc="upper left", fontsize=15)
    plt.savefig("{}.cumrew.png".format(task_name), bbox_inches="tight")
    plt.close()


def plot_CTR(task_name, df_CTR):
    plt.figure(figsize=(7, 7))
    plt.ylim([0.0, 1.0])
    plt.title("CTR (window=50) ({})".format(task_name), fontsize=25)
    plt.xlabel("Rounds (t)", fontsize=25)
    plt.ylabel("CTR", fontsize=25)
    for j in range(df_CTR.shape[1]):
        y = df_CTR.iloc[:, j]
        plt.plot(np.arange(df_CTR.shape[0]), y, label=df_CTR.columns[j])
    plt.legend(loc="upper left", fontsize=15)
    plt.savefig("{}.CTR.png".format(task_name), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
