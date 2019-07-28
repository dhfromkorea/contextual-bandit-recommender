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
    Ask for n samples of mushrooms.

    Parameters
    ----------
    n_mushrooms : int
                  The number of mushrooms to return.
    r_eat_good  : float
                  The reward for eating a good mushroom.
    r_eat_bad_lucky : float
                  The reward for eating a bad mushroom.
                  Then, come out a lucky break.
    r_eat_bad_unlucky : float
                  The penality for eating a bad mushroom.
    r_eat_bad_lucky_prob: float
                  The probability of avoiding the penalty.
    r_no_eat : float
                  The reward for not eating.

    Returns
    -------
    contexts : numpy array
                  Mushroom contexts.
    r_acts : numpy array of size (n_mushrooms, n_actions)
                  A sampled reward table.
    opt_acts_hidden : numpy array
                  Optimal actions.
    is_poisonous_hidden: numpy array
                  Whether each mushroom is bad.

    """
    n, d = X.shape
    indices = np.random.choice(np.arange(n), size=n_mushrooms)

    contexts = X[indices, :]

    # hidden info for the oracle (described below)
    is_poisonous_hidden = y[indices]
    is_edible_hidden = (~y[indices].astype(bool)).astype(int)

    r_no_eats = r_no_eat * np.ones(n_mushrooms)
    r_eats = np.zeros(n_mushrooms)
    # y_i == 1 => bad
    r_eats += r_eat_good * is_edible_hidden
    # y_i == 0 ==> good
    r_eat_bad = np.random.choice([r_eat_bad_lucky, r_eat_bad_unlucky],
                                 p=[r_eat_bad_lucky_prob, 1-r_eat_bad_lucky_prob],
                                 size=n_mushrooms)
    r_eats += r_eat_bad * is_poisonous_hidden

    # E[R_{t,a} | A_t=1, C_t=c_t, P_t=1]

    # assumes an oracle who has access to is_poisnous (hidden)
    # E[R_t,a | no_eat, good] = E[R_t,a | no_eat, bad]
    # E[R_t,a | eat, bad], E[R_t,a | eat, good]
    # assume E[R_t,a | eat, good] > rest
    E_r_no_eat = r_no_eat
    E_r_eat_bad = r_eat_bad_lucky * r_eat_bad_lucky_prob +\
                  r_eat_bad_unlucky * (1 - r_eat_bad_lucky_prob)

    if E_r_no_eat > E_r_eat_bad:
        # eat good
        # not eat bad
        opt_acts_hidden = is_edible_hidden
    else:
        # take risks: always eat
        opt_acts_hidden = np.ones(n_mushrooms, dtype=int)

    # hidden info, not to be given to an agent
    r_acts = np.hstack((r_no_eats[:, None], r_eats[:, None]))

    return contexts, r_acts, opt_acts_hidden, is_poisonous_hidden
