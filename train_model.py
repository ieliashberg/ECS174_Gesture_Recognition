import pandas as pd
import torch
import torchvision
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def load_and_split():
    df = pd.read_csv("static_dataset_scaled.csv")
    features = df.iloc[:, :-1].values.astype(np.float32)
    labels = df.iloc[:, -1].values  # currently still string labels
    encoder = LabelEncoder()    # create instance of the label encoder
    labels_encoded = encoder.fit_transform(labels)  #actually encode the string labels
    features_train, features_temp, labels_train, labels_temp = train_test_split(features, labels_encoded, test_size = 0.3, stratify = labels_encoded, random_state = 42) # split the data 70/30 into train/temp
    features_test, features_validation, labels_test, labels_validation = train_test_split(features_temp, labels_temp, test_size=0.5, stratify=labels_temp, random_state = 42) # split the temp into 50/50 train/validation

    return features_train, features_validation, features_test, labels_train, labels_validation, labels_test, encoder


def train_model(features_train, features_validation, labels_train, labels_validation):
    train_ds = TensorDataset(torch.from_numpy(features_train).float(),
                         torch.from_numpy(labels_train).long())
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    validation_ds = TensorDataset(torch.from_numpy(features_validation).float(), torch.from_numpy(labels_validation).long())

    validation_loader = DataLoader(validation_ds, batch_size = 16, shuffle = False)

