import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from pprint import pprint as pp

import os
# @todo: allow multiple files to be read
news_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "v1"))

ARTICLE_POOL_LENGTH = 20
# ARTICLE_POOL remains the same

def parse_uv_event(user_visit):
    """
    user_visit: str
    """
    uv_event = {}



    n_features = 6

    tokens = user_visit.strip().split(" ")
    uv_event["timestamp"] = tokens[0]
    uv_event["article_id_displayed"] = tokens[1]
    uv_event["user_click"]  = tokens[2]
    user_marker = tokens[3]

    #print("{}, {}, {}, {}".format(timestamp, article_id_dp, user_click, marker))

    uv_event["user"] = {}
    user_id = user_marker[1:]

    if user_marker == "|user":
        uv_event["user"][user_id] = [None] * n_features

        for user_feature in tokens[4:10]:
            feature_id, feature_val = user_feature.split(":")
            uv_event["user"][user_id][int(feature_id)-1] = feature_val

    else:
        raise Exception("unexpected marker: {}".format(user_marker))

    i = 10

    uv_event["article"] = {}
    while i < len(tokens):
        article_marker = tokens[i]
        if article_marker[0] == "|" and article_marker[1:].isdigit():
            # assumes pos int
            article_id =  article_marker[1:]
            uv_event["article"][article_id] = [None] * n_features
            for article_feature in tokens[i+1:i+7]:
                feature_id, feature_val = article_feature.split(":")
                uv_event["article"][article_id][int(feature_id)-1] = feature_val
        else:
            raise Exception("unexpected marker: {}".format(article_marker))
        i += 7
    #pp("number of articles: {}".format(len(article_features.keys()))

    return uv_event


def read_user_event():
    path = os.path.join(news_data_path, "ydata-fp-td-clicks-v1_0.20090501")
    with open(path, "r+") as f:
        for line in f:
            yield line


def sample_user_event(n_samples=-1):
    """
    read line by line without loading all in memory
    """
    data = []
    if n_samples == -1:
        # sample all
        n_samples = np.infty

    reader = read_user_event()

    i = 0
    while True:
        if i < n_samples:
            try:
                line = reader.next()
                i += 1
            except Exception as e:
                print("Error reading user event file: {}, {}".format(e.msg, e.args))
            uv_event = parse_uv_event(line)
            data.append(uv_event)
        #pp(uv_event)
    return data
    #return contexts, r_acts, opt_acts_hidden, hidden




