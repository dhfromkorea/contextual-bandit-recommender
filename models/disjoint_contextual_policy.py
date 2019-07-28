"""
The policies that use user contexts.

The current implementations build disjoint models for each action.

The current implementations do not accept item (action) contexts.

@todo: combine this with action_context_policies.
       1. disjoint, 2. shared.
"""
import numpy as np
from scipy.stats import invgamma


class LinUCBPolicy(object):
    """
    Implementation of Li, et al. [1]

    Disjoint model for each action.

    For computational reasons, model updates are done periodically.

    [1]: http://rob.schapire.net/papers/www10.pdf
    """
    def __init__(self, n_actions, context_dim, delta=0.2,
                 train_starts_at=500, train_freq=50):
        self._n_actions = n_actions
        self._act_count = np.zeros(n_actions, dtype=int)

        # bias
        self._d = context_dim + 1

        # initialize with I_d, 0_d
        self._A = [
                np.identity(self._d)
                for _ in range(self._n_actions)
        ]
        # for computational reasons
        # we update every train_freq
        self._A_inv = np.linalg.inv(self._A)

        self._b = [
                np.zeros(self._d)
                for _ in range(self._n_actions)
        ]

        self._theta = [
                self._A_inv[j].dot(self._b[j])
                for j in range(self._n_actions)
        ]

        self._alpha = 1 + np.sqrt(np.log(2/delta)/2)

        self._t = 0
        self._train_freq = train_freq
        self._train_starts_at = train_starts_at

    def choose_action(self, x_t):
        """

        solve X_a w_a + b_a = r_a

        where
        X_a in R^{n x d}
        b_a in R^{n x 1}
        D_a (n_j x d): design matrix for a_j
        theta (d x 1)

        solve D_a theta_a = c_a
        theta_a = (D_a^TD_a + I_d)^{-1}
        A_a = D_a^TD_a + I_d

        I_d perturbation singular
        X_a (d x d):

        alpha: constant coefficient for ucb
        at least 1 - delta probability
        alpha = 1 + sqrt(ln(2/delta)/2)
        copmute ucb

        assumes alpha given

        access to theta_a for all actions
        """
        # for bias
        x_t = np.append(x_t, 1)

        # estimate an action value
        Q = np.zeros(self._n_actions)
        ubc_t = np.zeros(self._n_actions)

        for j in range(self._n_actions):
            # compute upper bound
            k_ta = x_t.T.dot(self._A_inv[j]).dot(x_t)
            ubc_t[j] = self._alpha * np.sqrt(k_ta)
            Q[j] = self._theta[j].dot(x_t) + ubc_t[j]

        # todo: tiebreaking
        a_t = np.argmax(Q)

        self._t += 1

        return a_t

    def update(self, a_t, x_t, r_t):
        """
        """
        # d x 1
        x_t = np.append(x_t, 1)
        # d x d
        self._A[a_t] += x_t.dot(x_t.T)
        # d x 1
        self._b[a_t] += r_t * x_t

        if self._t < self._train_starts_at:
            return

        if self._t % self._train_freq == 0:
            # solve linear systems for one action
            # using lstsq to handle over/under determined systems
            self._theta[a_t] = self._A_inv[a_t].dot(self._b[a_t])


class LinearGaussianThompsonSamplingPolicy(object):
    """
    Linear Gaussian Thompson Sampling policy.

    A Bayesian approach for inferring the true reward distribution model.

    Implements a Bayesian linear regression with a conjugate prior [1].

    Model: Gaussian: R_t = W*x_t + eps, eps ~ N(0, sigma^2 I)
    Model Parameters: mu, cov
    Prior on the parameters
    p(w, sigma^2) = p(w|sigma^2) * p(sigma^2)

    1. p(sigma^2) ~ Inverse Gamma(a_0, b_0)
    2. p(w|sigma^2) ~ N(mu_0, sigma^2 * precision_0^-1)

    For computational reasons,

    1. model updates are done periodically.
    2. for large sample cases, batch_mode is enbabled.

    [1]: https://en.wikipedia.org/wiki/Bayesian_linear_regression#Conjugate_prior_distribution

    """

    def __init__(self,
                 n_actions,
                 context_dim,
                 eta_prior=6.0,
                 lambda_prior=0.25,
                 train_starts_at=500,
                 posterior_update_freq=50,
                 batch_mode=True,
                 batch_size=512,
                 lr=0.1):
        self._t = 1
        self._update_freq = posterior_update_freq
        self._train_starts_at = train_starts_at

        self._n_actions = n_actions
        # bias
        self._d = context_dim + 1

        # inverse gamma prior
        self._a_0 = eta_prior
        self._b_0 = eta_prior
        self._a_list = [eta_prior] * n_actions
        self._b_list = [eta_prior] * n_actions


        # conditional Gaussian prior
        self._sigma_sq_0 = invgamma.rvs(eta_prior, eta_prior)
        self._lambda_prior = lambda_prior
        # precision_0 shared for all actions
        self._precision_0 = self._sigma_sq_0 / self._lambda_prior * np.eye(self._d)

        # initialized at mu_0
        self._mu_list = [
                np.zeros(self._d)
                for _ in range(n_actions)
        ]

        # initialized at cov_0
        self._cov_list = [
                1.0 / self._lambda_prior * np.eye(self._d)
                for _ in range(n_actions)
        ]

        # remember training data
        self._train_data = [None] * n_actions

        # for computational efficiency
        # train on a random subset
        self._batch_mode = batch_mode
        self._batch_size = batch_size
        self._lr = lr


    def _update_posterior(self, act_t, X_t, r_t_list):
        cov_t = np.linalg.inv(np.dot(X_t.T, X_t) + self._precision_0)
        mu_t = np.dot(cov_t, np.dot(X_t.T, r_t_list))
        a_t = self._a_0 + self._t/2

        # mu_0 simplifies some terms
        r = np.dot(r_t_list, r_t_list)
        precision_t = np.linalg.inv(cov_t)
        b_t = self._b_0 + 0.5*(r - np.dot(mu_t.T, np.dot(precision_t, mu_t)))

        self._cov_list[act_t] = cov_t
        self._mu_list[act_t] = mu_t
        self._a_list[act_t] = a_t
        self._b_list[act_t] = b_t

        if self._batch_mode:
            # learn bit by bit
            self._cov_list[act_t] = cov_t * self._lr + \
                    self._cov_list[act_t] * (1 - self._lr)
            self._mu_list[act_t] = mu_t * self._lr + \
                    self._mu_list[act_t] * (1 - self._lr)
            self._a_list[act_t] = a_t * self._lr + \
                    self._a_list[act_t] * (1 - self._lr)
            self._b_list[act_t] = b_t * self._lr + \
                    self._b_list[act_t] * (1 - self._lr)
        else:
            self._cov_list[act_t] = cov_t
            self._mu_list[act_t] = mu_t
            self._a_list[act_t] = a_t
            self._b_list[act_t] = b_t


    def _sample_posterior_predictive(self, x_t, n_samples=1):
        # p(sigma^2)
        sigma_sq_t_list = [
            invgamma.rvs(self._a_list[j], scale=self._b_list[j])
            for j in range(self._n_actions)
        ]

        try:
            # p(w|sigma^2) = N(mu, sigam^2 * cov)
            W_t = [
                np.random.multivariate_normal(
                    self._mu_list[j] , sigma_sq_t_list[j] * self._cov_list[j]
                )
                for j in range(self._n_actions)
            ]
        except np.linalg.LinAlgError as e:
            print("Error in {}".format(type(self).__name__))
            print('Errors: {}.'.format(e.args[0]))
            W_t = [
                np.random.multivariate_normal(
                    np.zeros(self._d), np.eye(self._d)
                )
                for i in range(self._n_actions)
            ]

        # p(r_new | params)
        mean_t_predictive = np.dot(W_t, x_t)
        cov_t_predictive = sigma_sq_t_list * np.eye(self._n_actions)
        r_t_estimates = np.random.multivariate_normal(
                            mean_t_predictive,
                            cov=cov_t_predictive, size=1
                        )
        r_t_estimates = r_t_estimates.squeeze()

        assert r_t_estimates.shape[0] == self._n_actions

        return r_t_estimates


    def choose_action(self, x_t):
        x_t = np.append(x_t, 1)
        r_t_estimates = self._sample_posterior_predictive(x_t)
        act = np.argmax(r_t_estimates)

        self._t += 1

        return act


    def update(self, a_t, x_t, r_t):
        self._set_train_data(a_t, x_t, r_t)
        # sample model parameters
        # p(w, sigma^2 | X_t, r_vec_t)
        X_t, r_t_list = self._get_train_data(a_t)
        n_samples = X_t.shape[0]


        if self._t < self._train_starts_at:
            return

        # posterior update periodically per action
        if n_samples % self._update_freq == 0:
            self._update_posterior(a_t, X_t, r_t_list)


    def _get_train_data(self, a_t):
        return self._train_data[a_t]


    def _set_train_data(self, a_t, x_t, r_t):
        # add bias
        x_t = np.append(x_t, 1)

        if self._train_data[a_t] is None:
            X_t = x_t[None, :]
            r_t_list = np.array([r_t])

        else:
            X_t, r_t_list = self._train_data[a_t]
            n = X_t.shape[0]
            X_t = np.vstack((X_t, x_t))
            assert X_t.shape[0] == (n+1)
            assert X_t.shape[1] == self._d

            r_t_list = np.append(r_t_list, r_t)

        # train on a random batch
        n_samples = X_t.shape[0]
        if self._batch_mode and self._batch_size < n_samples:
            indices = np.arange(self._batch_size)
            batch_indices = np.random.choice(indices,
                                             size=self._batch_size,
                                             replace=False)
            X_t = X_t[batch_indices, :]
            r_t_list = r_t_list[batch_indices]

        self._train_data[a_t] = (X_t, r_t_list)
