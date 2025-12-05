# train_dynamic_model.py
import os
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import time

from ipn_dataset_util import NPZSequenceDatasetCPU, collate_np, augment_sequence
from utils import normalize

DATA_ROOT = Path("..\IPN\IPN_dynamic")
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
ADD_VEL = True
NUM_WORKERS = 4
PERSISTENT = True
PREFETCH = 2

def collate_np(batch):
    xs, ys, lens = zip(*batch)
    B = len(xs)
    D = xs[0].shape[1]
    lens = np.asarray(lens, dtype=np.int64)
    T_max = int(lens.max())
    x_pad = np.zeros((B, T_max, D), dtype=np.float32)
    for i, x in enumerate(xs):
        t = x.shape[0]
        x_pad[i, :t, :] = x
    y = np.asarray(ys, dtype=np.int64)
    return x_pad, lens, y

def worker_init(_):
    #Redefined the worker init so windows doesnt go ballistic
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

def load_and_split_npz(npz_root: Path, val_frac=0.15, test_frac=0.15, seed=1):
    files = sorted(npz_root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {npz_root}")

    labels = []
    for f in files:
        with np.load(f) as npz:
            labels.append(str(npz["label"]))

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        files, y, test_size=(val_frac + test_frac), stratify=y, random_state=seed
    )
    rel_test = test_frac / (val_frac + test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, encoder

def train_model(X_train, y_train, X_val, y_val, encoder, num_epochs=EPOCHS):
    #Imports are here for workers (windows gets mad if they aren't)
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from dynamic_gesture_net import DynamicGestureNet

    train_ds = NPZSequenceDatasetCPU(X_train, y_train, add_vel=ADD_VEL, normalizer=normalize, train = True, augmenter=augment_sequence)
    val_ds   = NPZSequenceDatasetCPU(X_val,   y_val,   add_vel=ADD_VEL, normalizer=normalize, train=False, augmenter=None)

    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_np, num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT, pin_memory=pin_mem,
        prefetch_factor=PREFETCH, worker_init_fn=worker_init
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_np, num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT, pin_memory=pin_mem,
        prefetch_factor=PREFETCH, worker_init_fn=worker_init
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    input_dim   = 63 * (2 if ADD_VEL else 1)
    num_classes = len(encoder.classes_)
    net = DynamicGestureNet(input_dim=input_dim, num_classes=num_classes,
                            hidden=128, layers=2, bidir=True, dropout=0.2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)

    training_accuracy, validation_accuracy = [], []

    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_loader), desc=f"epoch {epoch+1}", unit="batch")
        start = time.time()

        #Train from batch
        net.train()
        correct, total = 0, 0
        for X_np, lengths_np, y_np in train_loader:
            X = torch.from_numpy(X_np).to(device, non_blocking=pin_mem)
            lengths = torch.from_numpy(lengths_np).to(device, non_blocking=pin_mem)
            y = torch.from_numpy(y_np).to(device, non_blocking=pin_mem)

            optimizer.zero_grad()
            logits = net(X, lengths)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
            progress_bar.update(1)
        training_accuracy.append(correct / total if total else 0.0)

        #Validation
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_np, lengths_np, y_np in val_loader:
                X = torch.from_numpy(X_np).to(device, non_blocking=pin_mem)
                lengths = torch.from_numpy(lengths_np).to(device, non_blocking=pin_mem)
                y = torch.from_numpy(y_np).to(device, non_blocking=pin_mem)

                logits = net(X, lengths)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total   += y.size(0)
        validation_accuracy.append(correct / total if total else 0.0)

        best_val = 0.0
        patience, bad = 6, 0
        val_acc = validation_accuracy[-1] #checks the most recent validation accuracy for early stopping
        if val_acc > best_val:
            best_val = val_acc; bad = 0
            torch.save({"model_state_dict": net.state_dict(),
                        "input_dim": input_dim, "num_classes": num_classes,
                        "classes": encoder.classes_.tolist(), "add_vel": ADD_VEL},
                    "dynamic_gesture_net_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stop.")
                break

        progress_bar.close()
        end = time.time()
        print(f"epoch {epoch+1:02d} | train_acc={training_accuracy[-1]:.3f} | val_acc={validation_accuracy[-1]:.3f} | time={end - start}")

    dynamicGRU = {
        "model_state_dict": net.state_dict(),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "classes": encoder.classes_.tolist(),
        "add_vel": ADD_VEL,
    }
    torch.save(dynamicGRU, "dynamic_gesture_net_v3.pt")

    # Plot accuracies (same style)
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

    return training_accuracy, validation_accuracy

def test_model(X_test, y_test, dynamic_GRU="dynamic_gesture_net_v3.pt"):
    # Import torch only inside function
    import torch
    from torch.utils.data import DataLoader
    from dynamic_gesture_net import DynamicGestureNet

    test_ds = NPZSequenceDatasetCPU(X_test, y_test, add_vel=ADD_VEL, normalizer=normalize)

    def _worker_init(_):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    pin_mem = torch.cuda.is_available()
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_np, num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT, pin_memory=pin_mem,
        prefetch_factor=PREFETCH, worker_init_fn=worker_init
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    savedGRU = torch.load(dynamic_GRU, map_location=device)

    net = DynamicGestureNet(
        input_dim=savedGRU["input_dim"],
        num_classes=savedGRU["num_classes"],
        hidden=128, layers=2, bidir=True, dropout=0.3
    ).to(device)
    net.load_state_dict(savedGRU["model_state_dict"])
    net.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for X_np, lengths_np, y_np in test_loader:
            X = torch.from_numpy(X_np).to(device, non_blocking=pin_mem)
            lengths = torch.from_numpy(lengths_np).to(device, non_blocking=pin_mem)
            y = torch.from_numpy(y_np).to(device, non_blocking=pin_mem)

            logits = net(X, lengths)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    print(f"model test accuracy = {correct/total:.4f}")

def main():
    X_train, y_train, X_val, y_val, X_test, y_test, encoder = load_and_split_npz(DATA_ROOT)
    train_model(X_train, y_train, X_val, y_val, encoder, num_epochs=EPOCHS)
    test_model(X_test, y_test)

if __name__ == "__main__":
    main()
