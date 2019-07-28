"""
sample mushroom problem
according to:

https://arxiv.org/pdf/1802.09127.pdf

"""

import numpy as np


def sample_synthetic(n_samples, n_actions=5, context_dim=10, sigma=1.0):
    """
    W ~ samples hidden random Matrix of n_actions x context_dim
    x_t ~ samples contexts uniformly from {-5, -4, ..., 4, 5}
    r_t ~ samples rewards from an isotropic multivariate normal
          r_t ~ N(W x_t, \sigma^2 I)

    assumes no interaction between actions.

    returns:
        (n_samples)-length stream of data

    """

    np.random.seed(0)
    W = np.random.rand(n_actions, context_dim)
    cov = sigma**2 * np.eye(n_actions)

    x_t_list = []
    r_acts_list = []
    opt_act_list_hidden = []
    mean_list_hidden = []

    for t in range(n_samples):
        x_t = np.random.randint(low=-5, high=5, size=context_dim)
        mean = np.dot(W, x_t)

        r_acts = np.random.multivariate_normal(mean, cov=cov, size=1)
        r_acts = r_acts.squeeze()
        opt_act = np.argmax(mean)

        x_t_list.append(x_t)
        r_acts_list.append(r_acts)
        opt_act_list_hidden.append(opt_act)
        mean_list_hidden.append(mean)

    return x_t_list, r_acts_list, opt_act_list_hidden, mean_list_hidden




