"""
"""
import os

import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


datasets_dir = os.path.abspath(os.path.dirname(__file__))
print("datasets_dir", datasets_dir)
mushroom_data_path = os.path.join(datasets_dir, "mushroom/mushroom.csv")
print("mushroom_data_path", mushroom_data_path)


def load_data(name="mushroom"):
    if name == "mushroom":
        # do something
        df = pd.read_csv(mushroom_data_path)
        X, y = preprocess_data(df)

    else:
        raise Exception("undefined dataset")

    return X, y


def preprocess_data(df):
    """TODO: Docstring for preprocess_data.


    Parameters
    ----------
    arg1 : TODO

    Returns
    -------
    (X, y)

    X:
    y:

    """
    # preprocessing mushroom data
    df_ = pd.get_dummies(df.iloc[:, 1:])
    features, X = df_.columns, df_.values
    y = df.iloc[:, 0].values
    label_encoder_y = LabelEncoder()
    # y = 1 -> poisonous, y = 0 -> edible
    y = label_encoder_y.fit_transform(y)

    return X, y

