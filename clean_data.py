import mediapipe as mp
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

def make_translation_invariant(filename):
    df = pd.read_csv(filename)
    feature_cols = df.columns[:-1]               # all numeric features, exclude label
    # ensure float dtype for features
    df.loc[:, feature_cols] = df.loc[:, feature_cols].astype(float)

    # subtract x0 from every x_k (k = 0..20)
    df.loc[:, feature_cols[0::3]] = df.loc[:, feature_cols[0::3]].sub(df.loc[:, feature_cols[0]], axis=0)

    # subtract y0 from every y_k
    df.loc[:, feature_cols[1::3]] = df.loc[:, feature_cols[1::3]].sub(df.loc[:, feature_cols[1]], axis=0)

    # subtract z0 from every z_k
    df.loc[:, feature_cols[2::3]] = df.loc[:, feature_cols[2::3]].sub(df.loc[:, feature_cols[2]], axis=0)

    # write back to csv
    df.to_csv(filename, index=False)
    return df

# assuming already centered around wrist
def make_scale_invariant(filename):
    df = pd.read_csv(filename)
    feature_cols = df.columns[:-1]

    # ensure float dtype for features
    df.loc[:, feature_cols] = df.loc[:, feature_cols].astype(float)

    if not (np.allclose(df["x0"], 0, atol=1e-6) and np.allclose(df["y0"], 0, atol=1e-6) and np.allclose(df["z0"], 0, atol=1e-8)):
        print("not yet translation invariant. Please make translation invariant first")

    x_diffs = df["x9"] - df["x0"]
    y_diffs = df["y9"] - df["y0"]
    scale = np.sqrt(x_diffs**2 + y_diffs**2) + 1e-8 # avoid dividing by 0 so we add a tiny epsilon
    df.loc[:, feature_cols] = df.loc[:, feature_cols].div(scale, axis=0)

    df.to_csv(filename, index = False)
    return df

data_file = "static_dataset_copy.csv"
make_translation_invariant(data_file)
make_scale_invariant(data_file)
count_each_label(data_file)

