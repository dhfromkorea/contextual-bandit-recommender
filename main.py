import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# we should forget about small efficiencies
# say about 97% of the time
# premature optimization is the root of the evil
# yet, we should not pass up opportunities
# for the critical 3%


def sample_mushroom(X, y,
                    n_mushrooms,
                    r_eat_good=5.0,
                    r_eat_bad_lucky=5.0,
                    r_eat_bad_unlucky=-35.0,
                    r_eat_bad_lucky_prob=0.5,
                    r_no_eat=0.0
                    ):
    """
    takes:
        n_mushrooms
        r_eat_good =
        r_eat_bad_lucky =
        r_eat_bad_unlucky =
        r_eat_bad_luck_prob =

    ask for N mushrooms

    returns:
        (context, reward_eat, reward_no_eat), optimal actions
    """
    n, d = X.shape
    indices = np.random.choice(np.arange(n), size=n_mushrooms)

    contexts = X[indices, :]
    r_no_eats = r_no_eat * np.ones(n_mushrooms)
    r_eats = np.zeros(n_mushrooms)
    # y_i == 1 => good
    r_eats += r_eat_good * y[indices]
    # y_i == 0 ==> bad
    r_eat_bad = np.random.choice([r_eat_bad_lucky, r_eat_bad_unlucky],
                                 p=[r_eat_bad_lucky_prob, 1-r_eat_bad_lucky_prob],
                                 size=n_mushrooms)
    r_eats += r_eat_bad * ~y[indices].astype(bool)

    # E[R_{t,a} | A_t=1, C_t=c_t, P_t=1]


    # E[R_t,a | no_eat, good] = E[R_t,a | no_eat, bad]
    # = 0
    # E[R_t,a | eat, bad], E[R_t,a | eat, good]
    # =-15, 5
    # assume E[R_t,a | eat, good] > rest
    E_r_no_eat = r_no_eat
    E_r_eat_bad = r_eat_bad_lucky * r_eat_bad_lucky_prob +\
                  r_eat_bad_unlucky * (1 - r_eat_bad_lucky_prob)
    if E_r_no_eat > E_r_eat_bad:
        # not take risks
        # still eat when good
        opt_acts = y[indices]
    else:
        # take risks
        # always eat
        opt_acts = np.ones(n_mushrooms)



    return contexts, r_eats, r_no_eats, opt_acts




def main(arg1):
    """TODO: Docstring for main.

    Parameters
    ----------
    arg1 : TODO

    Returns
    -------
    TODO

    """

    # preprocessing mushroom data
    df = pd.read_csv("datasets/mushrooms.csv")
    df_ = pd.get_dummies(df.iloc[:, 1:])
    features, X = df_.columns, df_.values
    y = df.iloc[:, 0].values
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)


    # define a solver
    policy = random_policy

    # simulate the problem T steps
    T = 100
    mushrooms = sample_mushroom(X, y, T)
    cum_rewards = 0.0
    cum_regrets = 0.0
    t = 0

    # can regret be negative?
    for c_t, r_eat_t, r_no_eat_t, a_opt in zip(*mushrooms):
        a_t = policy(c_t)
        r_t = a_t * r_eat_t + (1 - a_t) * r_no_eat_t
        r_t_opt =  a_opt * r_eat_t + (1 - a_opt) * r_no_eat_t
        cum_rewards += r_t

        reg_t = r_t_opt - r_t
        cum_regrets += reg_t
        t += 1

    print("rewards")
    print(cum_rewards/T)

    print("regrets")
    print(cum_regrets/T)

def random_policy(c_t):
    return np.random.choice([0,1])

if __name__ == "__main__":
    args = []
    main(args)

