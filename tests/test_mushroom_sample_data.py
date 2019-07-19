import pytest
import numpy as np

from context import sample_mushroom
from context import load_data

@pytest.fixture
def mushroom_data():
    X, y = load_data(name="mushroom")
    return X, y


def test_mushroom_preprocessing_and_loading(mushroom_data):
    X, y = mushroom_data

    dim_context = 117
    n_samples = 8124

    assert X.shape[1] == dim_context
    assert X.shape[0] == n_samples
    assert len(y) == n_samples
    assert set(np.unique(y)) == set([0, 1])


def test_sample_mushroom(mushroom_data):
    X, y = mushroom_data

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

    contexts, r_eats, r_no_eats, opt_acts, is_poisonous = mushrooms

    dim_context = 117

    assert contexts.shape[0] == T
    assert contexts.shape[1] == dim_context

    assert np.mean(r_no_eats) == 0.0


    # good mush -> must eat
    is_edible  = ~(is_poisonous.astype(bool))
    assert np.all(opt_acts[is_edible])

