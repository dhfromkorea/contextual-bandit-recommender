import pytest
import numpy as np

from context import sample_mushroom

@pytest.fixture(scope="module")
def mushroom_data():
    # mock mushroom data
    dim_context = 117
    n_samples = 1000
    X = np.random.rand(n_samples, dim_context)
    y = np.random.choice([0, 1], size=n_samples)

    return X, y


def test_mushroom_preprocessing_and_loading(mushroom_data):
    X, y = mushroom_data

    dim_context = 117

    assert X.shape[1] == dim_context
    assert set(np.unique(y)) == set([0, 1])


def test_sample_mushroom(mushroom_data):
    X, y = mushroom_data

    n_samples = 5 * (10 ** 4)
    samples = sample_mushroom(X,
                              y,
                              n_samples,
                              r_eat_good=5.0,
                              r_eat_bad_lucky=5.0,
                              r_eat_bad_unlucky=-35.0,
                              r_eat_bad_lucky_prob=0.1,
                              r_no_eat=0.0
                              )

    contexts, r_acts, opt_acts_hidden, is_poisonous_hidden = samples
    r_no_eats, r_eats = r_acts[:, 0], r_acts[:, 1]

    dim_context = 117

    assert contexts.shape[0] == n_samples
    assert contexts.shape[1] == dim_context

    assert np.mean(r_no_eats) == 0.0


    # good mush -> must eat
    is_edible  = ~(is_poisonous_hidden.astype(bool))
    assert np.all(opt_acts_hidden[is_edible])

