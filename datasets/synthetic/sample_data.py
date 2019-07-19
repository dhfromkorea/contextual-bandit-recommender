"""
sample mushroom problem
according to:

https://arxiv.org/pdf/1802.09127.pdf

"""

import numpy as np


def sample_synthetic(n_samples, n_actions=5, context_dim=10, sigma=3.0):
    """
    takes:
        n_samples: i.i.d n contexts and other info

    parameters:
        r_acts: n_samples x n_actions numpy array
            each entry ij represent r | a_j, x_i

    we define a generator to be from
    a multivariate Bayesian linear regression model
    (cf https://en.wikipedia.org/wiki/Bayesian_multivariate_linear_regression)

    we assume no interactions between actions
    hence, we have a jointly independent model that gives:

    for each action j

    R_t^j = w_j*x_t + eps
    where eps _|_ x_t, eps ~ N(0, sigma_j^2)

    drop j for brevity

    parameters to eatimate: w, sigma^2

    define a joint prior distribution:
    p(w, sigma^2) = p(w|sigma)p(sigma)

    where p(sigma^2) ~ Inverse Gamma(a_0, b_0)
          p(w|sigma) ~ N(mu_0, sigma^2 * lambda_0^-1)

    lambda_0 = precision matrix
    lambda_0^-1 = co-variance matrix


    returns:
        context, r_acts, opt_acts, true_labels (if exists)
    """

    # define generating process
    # context dimension
    d = context_dim
    W = np.random.rand(n_actions, d)
    # independent covariance matrix
    # for convenience, isotropic gaussian
    cov = sigma**2 * np.eye(n_actions)

    for t in range(n_samples):
        # generate sparse context
        x_t = np.random.random_integers(low=0, high=5, size=d)
        mean = np.dot(W, x_t)

        r_acts = np.random.multivariate_normal(mean, cov=cov, size=1)
        r_acts = r_acts.squeeze()
        opt_act = np.argmax(mean)

        mean_hidden = mean

        yield x_t, r_acts, opt_act, mean_hidden




