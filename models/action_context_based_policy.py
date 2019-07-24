"""
"""
import numpy as np
from scipy.stats import invgamma


class SharedLinUCBPolicy(object):
    """
    action-context aware policy

    unlike LinUCB (disjoint) LinUCB (hybrid) in [1],
    all actions share all parameters.

    expected to underperform in a high data regime.
    expected to perform relatively well in a low data regime.

    Just a baseline.

    I did not like the idea of building a disjoint model
    for each action.

    [1]: https://arxiv.org/pdf/1003.0146.pdf
    """
    def __init__(self, n_actions, context_dim, delta=0.2,
                 train_starts_at=500, train_freq=50):
        self._n_actions = n_actions
        self._act_count = np.zeros(n_actions, dtype=int)

        # bias
        self._d = context_dim + 1

        # initialize with I_d, 0_d
        self._A = np.identity(self._d)

        self._b = np.zeros(self._d)

        self._theta = np.linalg.lstsq(self._A, self._b, rcond=None)[0]

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

        # estimate an action value
        Q = np.zeros(self._n_actions)
        ubc_t = np.zeros(self._n_actions)

        u_t, S_t = x_t
        n_actions = len(S_t)
        assert n_actions == self._n_actions
        assert len(S_t[0, :]) == self._d


        for j in range(self._n_actions):
            # compute input for each action
            # user_context + action_context + bias
            x_t = np.concatenate( (u_t, S_t[j, :], [1]) )

            # compute upper bound
            k_ta = x_t.T.dot(np.linalg.inv(self._A)).dot(x_t)
            ubc_t[j] = self._alpha * np.sqrt(k_ta)
            Q[j] = self._theta.dot(x_t) + ubc_t[j]

        # todo: tiebreaking
        a_t = np.argmax(Q)

        #if self._t % 10000 == 0:
        #    print("Q est {}".format(Q))
        #    print("ubc {} at {}".format(ubc_t[a_t], self._t))

        self._t += 1

        return a_t

    def update(self, a_t, x_t, r_t):
        """
        """

        u_t, S_t = x_t
        n_actions = len(S_t)
        assert n_actions == self._n_actions
        x_t = np.concatenate( (u_t, S_t[a_t, :], [1]) )
        assert len(x_t) == self._d

        # d x d
        self._A += x_t.dot(x_t.T)
        # d x 1
        self._b += r_t * x_t

        if self._t < self._train_starts_at:
            return

        if self._t % self._train_freq == 0:
            # solve linear systems for one action
            # using lstsq to handle over/under determined systems
            self._theta = np.linalg.lstsq(self._A, self._b, rcond=None)[0]


class SharedLinearGaussianThompsonSamplingPolicy(object):
    """

    action-context aware policy

    A variant of Thompson Sampling that is based on

    Bayesian Linear Regression

    with Inverse Gamma and Gaussian prior


    unlike the one in context_based_policy.py
    all parameters are shared across actions

    the performance expectation is similar as SharedLinUCBPolicy.

    """

    def __init__(self,
                 n_actions,
                 context_dim,
                 eta_prior=6.0,
                 lambda_prior=0.25,
                 train_starts_at=500,
                 posterior_update_freq=50):
        """
        a_0; location for IG t=0
        b_0; scale for IG t=0


        """

        self._t = 1
        self._update_freq = posterior_update_freq
        self._train_starts_at = train_starts_at

        self._n_actions = n_actions
        # bias
        self._d = context_dim + 1

        # inverse gamma prior
        self._a_0 = eta_prior
        self._b_0 = eta_prior
        self._a_list = eta_prior
        self._b_list = eta_prior


        # conditional Gaussian prior
        self._sigma_sq_0 = invgamma.rvs(eta_prior, eta_prior)
        self._lambda_prior = lambda_prior
        # precision_0 shared for all actions
        self._precision_0 = self._sigma_sq_0 / self._lambda_prior

        # initialized at mu_0
        self._mu_list = 0.0

        # initialized at cov_0
        self._cov_list = 1.0 / self._lambda_prior

        # remember training data
        self._train_data = None


    def _update_posterior(self, X_t, r_t_list):
        """
        p(w, sigma^2) = p(mu|cov)p(a, b)

        where p(sigma^2) ~ Inverse Gamma(a_0, b_0)
              p(w|sigma) ~ N(mu_0, sigma^2 * lambda_0^-1)

        does a full refit. online version could be implemented.

        """
        cov_t = np.linalg.inv(np.dot(X_t.T, X_t) + self._precision_0)
        mu_t = np.dot(cov_t, np.dot(X_t.T, r_t_list))
        a_t = self._a_0 + self._t/2

        # mu_0 simplifies some terms
        r = np.dot(r_t_list, r_t_list)
        precision_t = np.linalg.inv(cov_t)
        b_t = self._b_0 + 0.5*(r - np.dot(mu_t.T, np.dot(precision_t, mu_t)))

        self._cov_list = cov_t
        self._mu_list = mu_t
        self._a_list = a_t
        self._b_list = b_t


    def _sample_posterior_predictive(self, x_t, n_samples=1):
        """

        estimate

        p(R_new | X, R_old)
        = int p(R_new | params )p(params| X, R_old) d theta

        """

        # 1. p(sigma^2)
        sigma_sq_t = invgamma.rvs(self._a_list, scale=self._b_list)

        try:
            # p(w|sigma^2) = N(mu, sigam^2 * cov)
            w_t = np.random.multivariate_normal(
                    self._mu_list, sigma_sq_t_list * self._cov_list
            )
        except np.linalg.LinAlgError as e:
            print("Error in {}".format(type(self).__name__))
            print('Errors: {}.'.format(e.args[0]))
            w_t = np.random.multivariate_normal(
                    np.zeros(self._d), np.eye(self._d)
            )


        # modify context
        u_t, S_t = x_t
        n_actions = len(S_t)
        assert n_actions == self._n_actions
        x_ta = [
                np.concatenate( (u_t, S_t[j, :], [1]) )
                for j in range(self._n_actions)
        ]
        assert len(x_ta[0]) == self._d


        # 2. p(r_new | params)
        mean_t_predictive = [
                np.dot(w_t, x_ta[j])
                for j in range(self._n_actions)
        ]

        cov_t_predictive = sigma_sq_t * np.eye(self._n_actions)
        r_t_estimates = np.random.multivariate_normal(
                            mean_t_predictive,
                            cov=cov_t_predictive, size=1
                        )
        r_t_estimates = r_t_estimates.squeeze()

        assert r_t_estimates.shape[0] == self._n_actions

        return r_t_estimates


    def choose_action(self, x_t):
        # p(R_new | params_t)

        r_t_estimates = self._sample_posterior_predictive(x_t)
        act = np.argmax(r_t_estimates)

        self._t += 1

        return act


    def update(self, a_t, x_t, r_t):
        """
        @todo: unify c_t vs. x_t

        with respect to a_t
        """

        # add a new data point
        if self._X_t is None:
            X_t = x_t[None, :]
            r_t_list = [r_t]
        else:
            X_t, r_t_list = self._train_data[action_idx]
            n = X_t.shape[0]
            X_t = np.vstack( (X_t, x_t))
            assert X_t.shape[0] == (n+1)
            assert X_t.shape[1] == self._d
            r_t_list = np.append(r_t_list, r_t)

        # upate train data
        self._train_data = (X_t, r_t_list)

        # exploration phase to mitigate cold-start
        # quite ill-defined
        if self._t < self._train_starts_at:
            return

        # update posterior prioridically
        # for computational reasons
        n_samples = X_t.shape[0]
        if n_samples % self._update_freq == 0:
            self._update_posterior(X_t, r_t_list)

