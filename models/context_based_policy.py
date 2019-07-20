"""
"""
# context-based
# idea
#      - build a classifer and based on good/bad prediction, choose an optimal
#      - to have a fair game, apply online classification training
#      - but you cannot observe the true class
import numpy as np
from sklearn.linear_model import SGDRegressor
from scipy.stats import invgamma


class LinearRegressorPolicy(object):
    """
    trains SVM with SGD

    Q(a_t,c_t) estimates E[R_t | a_t, c_t]

    """
    def __init__(self, n_actions, loss="squared_loss", epsilon=0.1, alpha=1e-4):
        self._n_actions = n_actions
        self._clf = SGDRegressor(alpha=alpha, loss=loss)
        self._eps = epsilon
        self._alpha = alpha
        self._t = 0

    def choose_action(self, c_t):
        scores = []
        # escape cold start
        if self._t < 10:
            return np.random.choice(np.arange(self._n_actions))

        u = np.random.uniform()
        if u > self._eps:
            for a_j in range(self._n_actions):
                x_t = [a_j] + list(c_t)
                score = self._clf.predict(x_t)
                scores.append(score)

            a_t = np.argmax(scores)
        else:
            a_t = np.random.choice(np.arange(self._n_actions))


        # anneal eps
        self._eps = (1 - self._alpha)**self._t

        self._t += 1

        return a_t

    def update(self, a_t, c_t, r_t):
        """
        a_t: int
        c_t: list of int
        """
        # ignores context
        x_t = np.array([a_t] + list(c_t))[None, :]
        r_t = [r_t]
        self._clf.partial_fit(x_t, r_t)


class LinUCBPolicy(object):
    def __init__(self, n_actions, delta=0.05, train_freq=500):
        self._n_actions = n_actions
        self._act_count = np.zeros(n_actions, dtype=int)
        self._A = [None] * self._n_actions
        self._b = [None] * self._n_actions
        self._theta = [None] * self._n_actions

        self._alpha = 1 + np.sqrt(np.log(2/delta)/2)

        self._t = 0
        self._train_freq = train_freq

    def choose_action(self, c_t):
        """

        compose x_{t,a} = concat(a_t, c_t) for all actions
        solve X_a w_a = r_a

        where
        c_t in R^{d - 1 x 1}
        b_a in R^{m x 1}
        D_a (n_j x d): design matrix for a_j
        c_a (n_j x 1): rewards corresponding to D_a
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
        # @todo: remove j and consider adding bias

        x_t = [np.array([j] + list(c_t)) for j in range(self._n_actions)]

        d = len(c_t) + 1

        if self._t % self._train_freq == 0:
            # solve linear systems for all actions
            for j in range(self._n_actions):

                if self._A[j] is None:
                    self._A[j] = np.identity(d)
                    self._b[j] = np.zeros(d)

                self._theta[j] = np.linalg.lstsq(self._A[j], self._b[j], rcond=None)[0]



        # estimate an action value
        Q = np.zeros(self._n_actions)
        ubc_t = np.zeros(self._n_actions)

        for j in range(self._n_actions):
            # compute upper bound
            k_ta = x_t[j].T.dot(np.linalg.inv(self._A[j])).dot(x_t[j])
            ubc_t[j] = self._alpha * np.sqrt(k_ta)
            Q[j] = self._theta[j].dot(x_t[j]) + ubc_t[j]

        # todo: tiebreaking
        a_t = np.argmax(Q)

        #if self._t % 10000 == 0:
        #    print("Q est {}".format(Q))
        #    print("ubc {} at {}".format(ubc_t[a_t], self._t))

        self._t += 1

        return a_t

    def update(self, a_t, c_t, r_t):
        """
        """
        # d x 1
        x_t = np.array([a_t] + list(c_t))
        # d x d
        self._A[a_t] += x_t.dot(x_t.T)
        # d x 1
        self._b[a_t] += r_t * x_t


class LinUCBHybridPolicy(object):
    """Docstring for LinUCBHybridPolicy. """

    def __init__(self):
        """TODO: to be defined1. """
        raise NotImplementedError


class LinearGaussianThompsonSamplingPolicy(object):
    """

    a variant of Thompson Sampling
    that computes an exact posterior
    for a linear gaussian model
    with corresponding conjugate priors.

    Model: Gaussian Likelihood
           R_t = W*x_t + eps
           # eps ~ N(0, sigma^2 I)

    Model Parameters: mu, cov

    Prior on the parameters
    p(w, sigma^2) =

    p(sigma^2) ~ Inverse Gamma(a_0, b_0)
    * p(w|sigma^2) ~ N(mu_0, sigma^2 * precision_0^-1)

    The postestior update for this prior-likelihood combination
    can be done exactly in a closed form.
    i.e. Bayesian Linear Regression

    we maintain for each action j

    mu_t^j, cov_t^j, sigma_sq_t^j

    posterior updates can be done periodically (update_freq)
    or sequentially (not implemented)

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


    def _update_posterior(self, action_idx, X_t, r_t_list):
        """
        p(w, sigma^2) = p(mu|cov)p(a, b)

        where p(sigma^2) ~ Inverse Gamma(a_0, b_0)
              p(w|sigma) ~ N(mu_0, sigma^2 * lambda_0^-1)

        """
        cov_t = np.linalg.inv(np.dot(X_t.T, X_t) + self._precision_0)
        mu_t = np.dot(cov_t, np.dot(X_t.T, r_t_list))
        a_t = self._a_0 + self._t/2

        # mu_0 simplifies some terms
        r = np.dot(r_t_list, r_t_list)
        precision_t = np.linalg.inv(cov_t)
        b_t = self._b_0 + 0.5*(r - np.dot(mu_t.T, np.dot(precision_t, mu_t)))

        self._cov_list[action_idx] = cov_t
        self._mu_list[action_idx] = mu_t
        self._a_list[action_idx] = a_t
        self._b_list[action_idx] = b_t


    def _sample_posterior_predictive(self, x_t, n_samples=1):
        """

        estimate

        p(R_new | X, R_old)
        = int p(R_new | params )p(params| X, R_old) d theta

        """


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
            print('Errors: {} | {}.'.format(e.message, e.args))
            beta = [
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
        # p(R_new | params_t)

        # add bias

        x_t = np.append(x_t, 1)
        r_t_estimates = self._sample_posterior_predictive(x_t)
        act = np.argmax(r_t_estimates)

        self._t += 1

        return act


    def update(self, a_t, x_t, r_t):
        """
        @todo: unify c_t vs. x_t

        with respect to a_t
        """
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

    def _get_train_data(self, action_idx):
        return self._train_data[action_idx]


    def _set_train_data(self, action_idx, x_t, r_t):
        # add bias
        x_t = np.append(x_t, 1)

        if self._train_data[action_idx] is None:
            X_t = x_t[None, :]
            r_t_list = [r_t]

        else:
            X_t, r_t_list = self._train_data[action_idx]
            n = X_t.shape[0]
            X_t = np.vstack((X_t, x_t))
            assert X_t.shape[0] == (n+1)
            assert X_t.shape[1] == self._d

            r_t_list = np.append(r_t_list, r_t)

        self._train_data[action_idx] = (X_t, r_t_list)

