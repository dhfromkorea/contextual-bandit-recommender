"""
sample mushroom problem
according to:

https://arxiv.org/pdf/1802.09127.pdf

"""

import numpy as np

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
    # y_i == 1 => bad
    r_eats += r_eat_good * ~y[indices].astype(bool)
    # y_i == 0 ==> good
    r_eat_bad = np.random.choice([r_eat_bad_lucky, r_eat_bad_unlucky],
                                 p=[r_eat_bad_lucky_prob, 1-r_eat_bad_lucky_prob],
                                 size=n_mushrooms)
    r_eats += r_eat_bad * y[indices]

    # E[R_{t,a} | A_t=1, C_t=c_t, P_t=1]


    # E[R_t,a | no_eat, good] = E[R_t,a | no_eat, bad]
    # = 0
    # E[R_t,a | eat, bad], E[R_t,a | eat, good]
    # = 1, 5
    # assume E[R_t,a | eat, good] > rest
    E_r_no_eat = r_no_eat
    E_r_eat_bad = r_eat_bad_lucky * r_eat_bad_lucky_prob +\
                  r_eat_bad_unlucky * (1 - r_eat_bad_lucky_prob)
    if E_r_no_eat > E_r_eat_bad:
        # not take risks
        # still eat when good
        opt_acts = (~y[indices].astype(bool)).astype(int)
    else:
        # take risks
        # always eat
        opt_acts = np.ones(n_mushrooms)

    # hidden info, not to be given to an agent
    is_poisonous_hidden = y[indices]
    opt_acts_hidden = opt_acts

    return contexts, r_eats, r_no_eats, opt_acts_hidden, is_poisonous_hidden

