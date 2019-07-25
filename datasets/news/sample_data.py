import os
from subprocess import Popen


import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from pprint import pprint as pp

# @todo: allow multiple files to be read
news_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "v1"))

ARTICLE_POOL_LENGTH = 20
# ARTICLE_POOL remains the same

def parse_uv_event(user_visit):
    """
    user_visit: str
    """

    try:
        uv_event = {}

        n_features = 6

        tokens = user_visit.strip().split(" ")
        uv_event["timestamp"] = tokens[0]
        uv_event["displayed_article_id"] = int(tokens[1])
        uv_event["is_clicked"]  = int(tokens[2])
        user_marker = tokens[3]

        #print("{}, {}, {}, {}".format(timestamp, article_id_dp, user_click, marker))

        uv_event["user"] = {}

        if user_marker == "|user":
            uv_event["user"] = [None] * n_features

            for user_feature in tokens[4:10]:
                feature_id, feature_val = user_feature.split(":")
                uv_event["user"][int(feature_id)-1] = float(feature_val)

        else:
            raise Exception("unexpected marker: {}".format(user_marker))

        i = 10

        uv_event["article"] = {}
        while i < len(tokens):
            article_marker = tokens[i]
            if article_marker[0] == "|" and article_marker[1:].isdigit():
                # assumes pos int
                article_id =  int(article_marker[1:])
                uv_event["article"][article_id] = [None] * n_features
                for article_feature in tokens[i+1:i+7]:
                    feature_id, feature_val = article_feature.split(":")
                    uv_event["article"][article_id][int(feature_id)-1] = float(feature_val)
            else:
                raise Exception("unexpected marker: {}".format(article_marker))
            i += 7

        return uv_event

    except Exception as e:
        # print("Error while parsing {}\n{}".format(tokens, e.args[0]))
        # ignore this instance
        return None


def extract_data():
    from glob import glob
    compressed = glob(os.path.join(news_data_path, "*.gz"))
    for path in compressed:
        # if already extracted, ignore
        if os.path.exists(os.path.splitext(path)[0]):
            continue
        print("Extracting {}".format(os.path.basename(path)))
        cmd = ["gunzip", path]
        p = Popen(cmd)
        p.communicate()

        output_path = os.path.splitext(path)[0]
        cmd = ["mv", output_path, output_path + ".data"]
        p = Popen(cmd)
        p.communicate()


def read_user_event():
    from glob import glob
    paths = glob(os.path.join(news_data_path, "*.data"))

    for path in paths:
        if os.path.basename(path) == "ydata-fp-td-clicks-v1_0.20090501.data":
            # @todo: temp fix. remove this
            continue

    #path = os.path.join(news_data_path, "ydata-fp-td-clicks-v1_0.20090501.data")
        with open(path, "r+") as f:
            for line in f:
                yield line


def sample_user_event():
    """
    stream each line lazily

    each line represents a user visit event that may or may not be usable for
    a given policy

    note we don't have the fully-labeled (for all actions)
    """
    reader = read_user_event()

    for line in reader:
        uv = parse_uv_event(line)
        if uv is None:
            # corrupt data, ignore
            continue

        context_user = np.array(uv["user"])
        displayed_art_id = uv["displayed_article_id"]

        context_acts = np.array(list(uv["article"].values()))
        i = list(uv["article"]).index(displayed_art_id)
        r_acts = (i, int(uv["is_clicked"]))
        # useful for filtering data for training later on
        revealed_act_hidden = i

        yield context_user, context_acts, r_acts, revealed_act_hidden




