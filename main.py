import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class MushroomContextualBandit(object):
    """

    a wrapper (env) for contextual bandit problem
    - defines common states
    - defines common methods
        - receive a_t and observe r_t
        - play rounds

    """

    def __init__(self, mushrooms, replace=True):
        """
        """
        super().__init__()

        self._t = 0
        self._n_actions = 2
        self._replace = replace

        self._n_mushrooms = len(mushrooms)
        self._mushrooms = mushrooms
        self._mushrooms_no_replacement = \
                         np.random.permutation(self._n_mushrooms)


    def reset(self):
        self._t = 0

        self._mushrooms_no_replacement = \
                         np.random.permutation(self._n_mushrooms)
        # first context c_1
        c_t = self._sample_mushroom_context()
        return c_t


    def step(self, a_t):
        if self._t > self._n_mushrooms:
            raise Exception("No more mushrooms left")

        r_t = self._compute_reward(a_t)
        c_t = self._sample_mushroom_context()

        self._t += 1

        return c_t, r_t


    def _compute_reward(self, a_t):
        """
        """

        # hard code for now
        reward_eat_good = 5.0
        reward_eat_bad = -35.0
        reward_no_eat = 0.0

        if a_t == 1:
            # if eat
            if self._p_t:
                rewards = [reward_eat_good, reward_eat_bad]
                probs = [0.5, 0.5]
                r_t = np.random.choice(rewards, p=probs)
            else:
                r_t = reward_eat_good
        else:
            r_t = reward_no_eat
        return r_t


    def _sample_mushroom_context(self):
        """
        """
        if self._replace:
            m_t = np.random.choice(self._mushrooms)
        else:
            m_t = self._mushrooms[self._mushrooms_no_replacement[self._t]]

        self._p_t = m_t[0]
        self._c_t = m_t[1:]
        return self._c_t



def main(arg1):
    """TODO: Docstring for main.

    Parameters
    ----------
    arg1 : TODO

    Returns
    -------
    TODO

    """

    # train, test split
    # crossvalidation
    # model

    df = pd.read_csv("datasets/mushrooms.csv")
    df_ = pd.get_dummies(df.iloc[:, 1:])
    features, X = df_.columns, df_.values
    y = df.iloc[:, 0].values
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)

    # mushrooms
    M = np.concatenate((y[:, np.newaxis], X), axis=1)

    # define mushroom problem
    env = MushroomContextualBandit(M, replace=False)

    T = 50

    c_t = env.reset() # c_1

    for t in range(T):
        a_t = policy(c_t)
        c_t, r_t = env.step(a_t)
        assert isinstance(r_t, float)
        print("context_t: {}".format(c_t))
        print("reward_t: {}".format(r_t))
        #policy.update(c_t, a_t, r_t)

def policy(c_t):
    return 1

if __name__ == "__main__":
    args = []
    main(args)

