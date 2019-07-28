"""
"""
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder


datasets_dir = os.path.abspath(os.path.dirname(__file__))
mushroom_data_path = os.path.join(datasets_dir, "mushroom/mushroom.csv")


def load_data(name="mushroom"):
    if name == "mushroom":
        # do something
        df = pd.read_csv(mushroom_data_path)
        X, y = preprocess_data(df)
    else:
        raise Exception("Undefined dataset {}".format(name))

    return X, y


def preprocess_data(df):
    df_ = pd.get_dummies(df.iloc[:, 1:])
    _, X = df_.columns, df_.values
    y = df.iloc[:, 0].values
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)

    return X, y

