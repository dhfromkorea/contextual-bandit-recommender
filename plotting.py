import os


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("husl")

root_dir = os.path.abspath(os.path.dirname(__file__))

def main():
    tasks = ["mushroom", "synthetic"]
    for task in tasks:
        cumreg_path = os.path.join(root_dir, "{}.cumreg.csv".format(task))
        acts_path = os.path.join(root_dir, "{}.acts.csv".format(task))

        df_cumreg = pd.read_csv(cumreg_path)
        df_acts = pd.read_csv(acts_path)

        plot_cumreg(task, df_cumreg)
        plot_acts(task, df_acts)

def plot_cumreg(task_name, df_cumreg):
    plt.figure(figsize=(10, 10))
    for j in range(df_cumreg.shape[1]):
        y = df_cumreg.iloc[:, j]
        plt.plot(np.arange(df_cumreg.shape[0]), y, label=df_cumreg.columns[j])
    plt.legend(fontsize=20)
    plt.savefig("{}.cumreg.png".format(task_name), bbox_inches="tight")
    plt.close()


def plot_acts(task_name, df_acts):
    fig = plt.figure(figsize=(10, 10))
    for j in range(df_acts.shape[1]):
        y = df_acts.iloc[:, j]
        plt.scatter(np.arange(df_acts.shape[0]), y + j * 0.05, label=df_acts.columns[j])
    plt.yticks(np.arange(df_acts.shape[1]))
    plt.legend(bbox_to_anchor=(1,0), loc="lower right",
              bbox_transform=fig.transFigure, ncol= df_acts.shape[1], fontsize=20)
    plt.savefig("{}.acts.png".format(task_name), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
