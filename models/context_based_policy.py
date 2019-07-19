"""
"""
# context-based
# idea
#      - build a classifer and based on good/bad prediction, choose an optimal
#      - to have a fair game, apply online classification training
#      - but you cannot observe the true class
import numpy as np
from sklearn.linear_model import SGDRegressor
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
        c_t \in R^{d - 1 x 1}
        b_a \in R^{m x 1}
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

