import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from model_definitions.static_gesture_net import StaticGestureNet
from utils import normalize_data_file


def load_and_split(filename):
    normalize_data_file(filename)
    df = pd.read_csv(filename)
    features = df.iloc[:, :-1].values.astype(np.float32)
    labels = df.iloc[:, -1].values
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    features_train, features_temp, labels_train, labels_temp = train_test_split(
        features,
        labels_encoded,
        test_size=0.3,
        stratify=labels_encoded,
        random_state=42,
    )
    features_test, features_validation, labels_test, labels_validation = (
        train_test_split(
            features_temp,
            labels_temp,
            test_size=0.5,
            stratify=labels_temp,
            random_state=42,
        )
    )

    return (
        features_train,
        features_validation,
        features_test,
        labels_train,
        labels_validation,
        labels_test,
        encoder,
    )


def train_model(
    features_train,
    features_validation,
    labels_train,
    labels_validation,
    encoder,
    num_epochs,
):
    train_ds = TensorDataset(
        torch.from_numpy(features_train).float(), torch.from_numpy(labels_train).long()
    )
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    validation_ds = TensorDataset(
        torch.from_numpy(features_validation).float(),
        torch.from_numpy(labels_validation).long(),
    )

    validation_loader = DataLoader(validation_ds, batch_size=4, shuffle=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    net = StaticGestureNet(features_train.shape[1], len(encoder.classes_))
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    training_accuracy = []
    validation_accuracy = []
    for epoch in range(num_epochs):
        progress_bar = tqdm(
            total=len(train_loader), desc=f"epoch {epoch+1}", unit="batch"
        )

        net.train() #training using batchnorm and dropout
        num_correct_train = 0
        num_total_train = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)

            outputs = net(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(
                outputs, 1
            )
            num_correct_train += (
                (predicted == y).sum().item()
            )
            num_total_train += len(y)
            progress_bar.update(1)
        training_accuracy.append(num_correct_train / num_total_train)

        num_correct_validation = 0
        num_total_validation = 0
        with torch.no_grad():
            for X, y in validation_loader:
                net.eval()
                X, y = X.to(device), y.to(
                    device
                )
                outputs = net(X)
                _, predicted = torch.max(
                    outputs, 1
                )
                num_correct_validation += (predicted == y).sum().item()
                num_total_validation += len(y)
        validation_accuracy.append(num_correct_validation / num_total_validation)
    progress_bar.close()

    checkpoint = {
        "model_state_dict": net.state_dict(),
        "input_dim": features_train.shape[1],
        "num_classes": len(encoder.classes_),
        "classes": encoder.classes_.tolist(),
    }
    torch.save(checkpoint, "static_gesture_net_v1.pt")

    epochs = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, training_accuracy, label="Training Accuracy")
    plt.plot(epochs, validation_accuracy, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
    print(validation_accuracy)

def test_model(features_test, labels_test):
    test_ds = TensorDataset(
        torch.from_numpy(features_test).float(), torch.from_numpy(labels_test).long()
    )
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = torch.load("static_gesture_net_v1.pt", map_location="cpu")
    net = StaticGestureNet(model["input_dim"], model["num_classes"])
    net.load_state_dict(state_dict=model["model_state_dict"])

    num_test_correct = 0
    num_total_test = 0
    with torch.no_grad():
        for X, y in test_loader:
            net.eval()
            X, y = X.to(device), y.to(
                device
            )
            outputs = net(X)
            _, predicted = torch.max(
                outputs, 1
            )
            num_test_correct += (predicted == y).sum().item()
            num_total_test += len(y)
    print(f"model test accuracy = {num_test_correct/num_total_test}")

def main():
    (
        features_train,
        features_validation,
        features_test,
        labels_train,
        labels_validation,
        labels_test,
        encoder,
    ) = load_and_split("datasets/static_dataset.csv")
    train_model(
        features_train,
        features_validation,
        labels_train,
        labels_validation,
        encoder,
        30,
    )
    test_model(
        features_test,
        labels_test
    )

if __name__ == "__main__":
    main()
