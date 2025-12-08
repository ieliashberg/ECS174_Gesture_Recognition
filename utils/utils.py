import csv
import pandas as pd
import numpy as np

def count_each_label(filename):
    #dictionary storing how many labels there are
    label_to_count = {}

    #count how many of each gesture there are in the dataset (want there to be all roughly the same amount)
    with open(filename, "r", newline = "") as f:
        reader = csv.reader(f)
        label_to_count = {}
        for row in reader:
            label = row[-1]
            if label == "label":
                continue
            if row[-1] in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        print(label_to_count)

def make_translation_invariant(features):
    features = features.astype(float)

    #Columns
    xs = np.arange(0, features.shape[1], 3)
    ys = xs + 1
    zs = xs + 2

    #subtract wrist coordinates
    features[:, xs] -= features[:, xs[0], None]
    features[:, ys] -= features[:, ys[0], None]
    features[:, zs] -= features[:, zs[0], None]

    return features

#This should be run after wrist normalization
def make_scale_invariant(features):

    features = features.astype(float)

    xs = np.arange(0, features.shape[1], 3)
    ys = xs + 1
    zs = xs + 2

    #anchor to writst
    x0 = features[:, xs[0]]
    y0 = features[:, ys[0]]
    z0 = features[:, zs[0]]

    #reference joint 9
    x9 = features[:, xs[9]]
    y9 = features[:, ys[9]]
    z9 = features[:, zs[9]]

    scale = np.sqrt((x9 - x0)**2 + (y9 - y0)**2 + (z9-z0)**2) + 1e-8

    features /= scale[:, None]

    return features

def normalize(features):
    features = np.asarray(features, dtype=float)

    single_sample = False
    if features.ndim == 1:
        features = features[None, :]
        single_sample = True

    features = make_translation_invariant(features)
    features = make_scale_invariant(features)

    if single_sample:
        return features[0]
    return features

def normalize_data_file(filename):
    df: pd.DataFrame = pd.read_csv(filename)
    features = df.iloc[:, :-1].to_numpy()
    labels =df.iloc[:, -1].to_numpy()

    features = normalize(features)

    df.iloc[:, :-1] = features
    df.iloc[:, -1] = labels

    df.to_csv(filename, index = False)

