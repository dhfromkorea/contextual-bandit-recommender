import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pprint import pprint as pp


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
    # y = 1 -> poisonous, y = 0 -> edible
    y = label_encoder_y.fit_transform(y)


    # define a solver
    rp = RandomPolicy(n_actions=2)
    smp = SampleMeanPolicy(n_actions=2, lr=0.1)
    egp = EpsilonGreedyPolicy(n_actions=2, lr=0.1, epsilon=0.1)
    ucbp = UCBPolicy(n_actions=2, lr=0.01)
    lrp = LinearRegressorPolicy(n_actions=2)

    policies = [rp, smp, egp, ucbp]
    results = {}


    # simulate the problem T steps
    T = 5 * (10 ** 4)
    mushrooms = sample_mushroom(X,
                                y,
                                T,
                                r_eat_good=5.0,
                                r_eat_bad_lucky=5.0,
                                r_eat_bad_unlucky=-35.0,
                                r_eat_bad_lucky_prob=0.5,
                                r_no_eat=0.0
                                )

    for i, policy in enumerate(policies):

        rewards = np.zeros(T)
        regrets = np.zeros(T)
        t = 0

        # can regret be negative?
        for c_t, r_eat_t, r_no_eat_t, a_opt in zip(*mushrooms):
            a_t = policy.choose_action(c_t)
            r_t = a_t * r_eat_t + (1 - a_t) * r_no_eat_t

            policy.update(a_t, c_t, r_t)

            r_t_opt =  a_opt * r_eat_t + (1 - a_opt) * r_no_eat_t

            rewards[t] = r_t
            regrets[t] = r_t_opt - r_t

            t += 1

        results[i] = (np.mean(rewards), np.mean(regrets))

    print(results)


# context-free

# no exploration
class RandomPolicy(object):
    def __init__(self, n_actions):
        self._n_actions = n_actions

    def choose_action(self, c_t):
        return np.random.choice(np.arange(self._n_actions))

    def update(self, a_t, c_t, r_t):
        pass


# no exploration
class SampleMeanPolicy(object):
    def __init__(self, n_actions, lr=0.1):
        self._Q = np.zeros(n_actions)
        self._act_count = np.zeros(n_actions)
        self._lr = lr

    def choose_action(self, c_t):
        a_t = np.argmax(self._Q)
        self._act_count[a_t] += 1
        return a_t

    def update(self, a_t, c_t, r_t):
        # ignores context
        n_j = self._act_count[a_t]
        self._Q[a_t] = 1/n_j * (self._Q[a_t] *(n_j - 1) + r_t)


# exploration (no annealing)
class EpsilonGreedyPolicy(object):
    def __init__(self, n_actions, lr=0.1, epsilon=0.1):
        self._n_actions = n_actions
        self._Q = np.zeros(n_actions)
        self._act_count = np.zeros(n_actions)
        self._lr = lr
        self._eps = epsilon
        self._t = 0

    def choose_action(self, c_t):
        u = np.random.uniform()
        if u > self._eps:
            a_t = np.argmax(self._Q)
        else:
            # choose random
            a_t = np.random.choice(np.arange(self._n_actions))
        self._act_count[a_t] += 1

        # anneal eps
        self._eps = (1 - self._lr)**self._t

        self._t += 1
        return a_t

    def update(self, a_t, c_t, r_t):
        # ignores context
        n_j = self._act_count[a_t]
        self._Q[a_t] = 1/n_j * (self._Q[a_t] *(n_j - 1) + r_t)

# exploration (annealing)
class UCBPolicy(object):
    def __init__(self, n_actions, lr=0.1):
        self._n_actions = n_actions
        self._Q = np.zeros(n_actions)
        self._act_count = np.zeros(n_actions, dtype=int)
        self._lr = lr
        self._t = 1

    def choose_action(self, c_t):
        ubc_t = np.zeros(self._n_actions)
        for a_j in range(self._n_actions):
            # compute upper bound
            n_j = self._act_count[a_j]
            if n_j == 0:
                ubc_t[a_j] = np.inf
            else:
                ubc_t[a_j] = np.sqrt(2*np.log(self._t)/n_j)


        if self._t % 10000 == 0:
            print("ubc {} at {}".format(ubc_t, self._t))
            print("act count", self._act_count)
        a_t = np.argmax(self._Q + ubc_t)
        self._act_count[a_t] += 1
        self._t += 1
        return a_t

    def update(self, a_t, c_t, r_t):
        # ignores context
        n_j = self._act_count[a_t]
        self._Q[a_t] = 1/n_j * (self._Q[a_t] *(n_j - 1) + r_t)
        #self._Q[a_t] = self._Q[a_t] + self._lr * (r_t - self._Q[a_t])



# context-based
# idea
#      - build a classifer and based on good/bad prediction, choose an optimal
#      - to have a fair game, apply online classification training
#      - but you cannot observe the true class

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
        x_t = [a_t] + list(c_t)
        self._clf.partial_fit(x_t, r_t, classes=np.arange(self._n_actions))


if __name__ == "__main__":
    args = []
    main(args)

