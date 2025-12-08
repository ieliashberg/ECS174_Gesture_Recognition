# train_dynamic_model.py
import os
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.ipn_dataset_util import NPZSequenceDatasetCPU, augment_sequence

DATA_ROOT   = Path("datasets/IPN/IPN_dynamic_npz_normalized")
SAVE_DIR    = Path("dynamic_gesture_net_v11.pt")
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 0.001
ADD_VEL     = False
NUM_WORKERS = 6
PERSISTENT  = True
PREFETCH    = 2
SPLIT       = 0.3

def worker_init(_):
    #redefined the worker init so that windows doesnt scream at me
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def collate_torch(batch):
    import torch

    xs, ys, lens = zip(*batch)
    lens = np.asarray(lens, dtype=np.int64)
    B = len(xs)
    T_max = int(lens.max())
    D = xs[0].shape[1]

    x_pad = torch.zeros((B, T_max, D), dtype=torch.float32)
    for i, x in enumerate(xs):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        t = x.shape[0]
        x_pad[i, :t, :] = x

    lengths = torch.from_numpy(lens.astype(np.int64))
    labels  = torch.from_numpy(np.asarray(ys, dtype=np.int64))
    return x_pad, lengths, labels

def load_and_split_npz(npz_root: Path, val_frac=(SPLIT/2), test_frac=(SPLIT/2), seed=1):
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
    #Imports are so that workers can do multiprocessing
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from model_definitions.dynamic_gesture_net_with_attention import DynamicGestureNet #Remember to change test model too

    train_ds = NPZSequenceDatasetCPU(
        X_train, y_train,
        add_vel=ADD_VEL,
        train=True,
        augmenter=augment_sequence,
    )
    val_ds = NPZSequenceDatasetCPU(
        X_val, y_val,
        add_vel=ADD_VEL,
        train=False,
        augmenter=None,
    )

    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_torch,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT,
        pin_memory=pin_mem,
        prefetch_factor=PREFETCH,
        worker_init_fn=worker_init,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_torch,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT,
        pin_memory=pin_mem,
        prefetch_factor=PREFETCH,
        worker_init_fn=worker_init,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    input_dim   = 63 * (2 if ADD_VEL else 1)
    num_classes = len(encoder.classes_)

    net = DynamicGestureNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden=128,
        layers=2,
        bidir=True,
        dropout=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)

    training_accuracy   = []
    validation_accuracy = []

    best_val = -1.0
    patience = 10
    bad = 0

    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_loader), desc=f"epoch {epoch+1}", unit="batch")
        start = time.time()
        
        net.train()
        correct, total = 0, 0
        for X, lengths, y in train_loader:
            X = X.to(device, non_blocking=pin_mem)
            lengths = lengths.to(device, non_blocking=pin_mem)
            y = y.to(device, non_blocking=pin_mem)

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

        train_acc = correct / total if total else 0.0
        training_accuracy.append(train_acc)
        
        #validation
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, lengths, y in val_loader:
                X = X.to(device, non_blocking=pin_mem)
                lengths = lengths.to(device, non_blocking=pin_mem)
                y = y.to(device, non_blocking=pin_mem)

                logits = net(X, lengths)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total   += y.size(0)

        val_acc = correct / total if total else 0.0
        validation_accuracy.append(val_acc)
        
        #simple early stopping process
        if val_acc > best_val:
            best_val = val_acc
            bad = 0
            torch.save(
                {
                    "model_state_dict": net.state_dict(),
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "classes": encoder.classes_.tolist(),
                    "add_vel": ADD_VEL,
                },
                "dynamic_gesture_net_best.pt",
            )
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {epoch+1}. Best val_acc={best_val:.4f}")
                progress_bar.close()
                break

        progress_bar.close()
        end = time.time()
        print(
            f"epoch {epoch+1:02d} | "
            f"train_acc={train_acc:.3f} | "
            f"val_acc={val_acc:.3f} | "
            f"time={end - start:.1f}s"
        )
        
    dynamicGRU = {
        "model_state_dict": net.state_dict(),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "classes": encoder.classes_.tolist(),
        "add_vel": ADD_VEL,
    }
    torch.save(dynamicGRU, SAVE_DIR)

    #Plot accuracies
    epochs = range(1, len(training_accuracy) + 1)
    plt.figure()
    plt.plot(epochs, training_accuracy, label="Training Accuracy")
    plt.plot(epochs, validation_accuracy, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Validation accuracies:", validation_accuracy)

    return training_accuracy, validation_accuracy

def test_model(X_test, y_test, dynamic_GRU=SAVE_DIR):
    import torch
    from torch.utils.data import DataLoader
    from model_definitions.dynamic_gesture_net_with_attention import DynamicGestureNet #<<----ILAN---------------------------------------
    '''Make sure to change this load to the same model used for training or it will throw an error in net.load_state_dict'''

    test_ds = NPZSequenceDatasetCPU(X_test, y_test, add_vel=ADD_VEL, train=False, augmenter=None)

    pin_mem = torch.cuda.is_available()
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_torch,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT,
        pin_memory=pin_mem,
        prefetch_factor=PREFETCH,
        worker_init_fn=worker_init,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loadGRU = torch.load(dynamic_GRU, map_location=device)

    net = DynamicGestureNet(
        input_dim=loadGRU["input_dim"],
        num_classes=loadGRU["num_classes"],
        hidden=128,
        layers=2,
        bidir=True,
        dropout=0.2,
    ).to(device)
    net.load_state_dict(loadGRU["model_state_dict"])
    net.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for X, lengths, y in test_loader:
            X = X.to(device, non_blocking=pin_mem)
            lengths = lengths.to(device, non_blocking=pin_mem)
            y = y.to(device, non_blocking=pin_mem)

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
