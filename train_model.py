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
from static_gesture_net import StaticGestureNet


def load_and_split(filename):
    df = pd.read_csv(filename)
    features = df.iloc[:, :-1].values.astype(np.float32)
    labels = df.iloc[:, -1].values  # currently still string labels
    encoder = LabelEncoder()  # create instance of the label encoder
    labels_encoded = encoder.fit_transform(labels)  # actually encode the string labels
    features_train, features_temp, labels_train, labels_temp = train_test_split(
        features,
        labels_encoded,
        test_size=0.3,
        stratify=labels_encoded,
        random_state=42,
    )  # split the data 70/30 into train/temp
    features_test, features_validation, labels_test, labels_validation = (
        train_test_split(
            features_temp,
            labels_temp,
            test_size=0.5,
            stratify=labels_temp,
            random_state=42,
        )
    )  # split the temp into 50/50 train/validation

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
    )  # use GPU for training if available. Otherwise CPU

    # create an instance of our neural network class
    net = StaticGestureNet(features_train.shape[1], len(encoder.classes_))
    net.to(device)  # move our network instance to the gpu if we have one

    criterion = nn.CrossEntropyLoss()  # set our loss function
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # set our optimizer

    training_accuracy = []
    validation_accuracy = []
    for epoch in range(num_epochs):
        progress_bar = tqdm(
            total=len(train_loader), desc=f"epoch {epoch+1}", unit="batch"
        )

        net.train()  # using batch norm and dropout
        num_correct_train = 0
        num_total_train = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)

            outputs = net(X)  # forward pass and calculates outputs
            loss = criterion(outputs, y)  # get cross entropy loss
            loss.backward()  # autograd calculates all the gradients for us
            optimizer.step()  # updates the model weights using the graidents and current model params

            _, predicted = torch.max(
                outputs, 1
            )  # get the argmax of ouputs and set that to predicted
            num_correct_train += (
                (predicted == y).sum().item()
            )  # sum the amount of correct
            num_total_train += len(y)
            progress_bar.update(1)  # update the progress bar
        training_accuracy.append(num_correct_train / num_total_train)

        num_correct_validation = 0
        num_total_validation = 0
        with torch.no_grad():
            for X, y in validation_loader:
                net.eval()  # turing off batch norm and dropout for evaulation
                X, y = X.to(device), y.to(
                    device
                )  # move inputs and labels to the actual device (gpu if using)
                outputs = net(X)  # get the 1d vector of likelihoods for each class
                _, predicted = torch.max(
                    outputs, 1
                )  # take the argmax of each class for each output and set that to predicted
                num_correct_validation += (predicted == y).sum().item()
                num_total_validation += len(y)
        validation_accuracy.append(num_correct_validation / num_total_validation)
    progress_bar.close()

    # after loop: write one file
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


# need to return the parameters of the model or save them somehow


def main():
    (
        features_train,
        features_validation,
        features_test,
        labels_train,
        labels_validation,
        labels_test,
        encoder,
    ) = load_and_split("static_dataset_copy.csv")
    train_model(
        features_train,
        features_validation,
        labels_train,
        labels_validation,
        encoder,
        30,
    )

if __name__ == "__main__":
    main()
