import os
import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
PROCESSED_DIR = Path("datasets/jester/processed_15frames")  # contains train/validation .npy files
TRAIN_LABELS = Path("datasets/jester/jester_labels/jester_train_labels.csv")
VALID_LABELS = Path("datasets/jester/jester_labels/jester_validation_labels.csv")

# Classes we want to plot
TARGET_CLASSES = {
    "Shaking Hand",
    "Sliding Two Fingers Down",
    "Sliding Two Fingers Up",
    "Zooming In With Full Hand",
    "Zooming Out With Full Hand",
    "Zooming In With Two Fingers",
    "Zooming Out With Two Fingers",
    "Drumming Fingers",
    "Doing other things",
    "Swiping Down", 
    "Swiping Up",
}


# -----------------------------------
# LOAD VIDEO → LABEL MAPPING
# -----------------------------------
def load_label_map(csv_path):
    video_to_label = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f, delimiter=";")
        for video_num, gesture in reader:
            video_to_label[video_num] = gesture
    return video_to_label


# Merge train + validation maps
video_to_label = {}
video_to_label.update(load_label_map(TRAIN_LABELS))
video_to_label.update(load_label_map(VALID_LABELS))


# -----------------------------------
# GATHER FRAME COUNTS PER CLASS
# -----------------------------------
class_to_lengths = {cls: [] for cls in TARGET_CLASSES}

# Search all processed npy/npz files
all_processed = list(PROCESSED_DIR.rglob("*.npy"))

for npy_path in all_processed:
    video_id = npy_path.stem  # e.g. "34870"

    # Skip if not labeled or not in target classes
    gesture = video_to_label.get(video_id)
    if gesture not in TARGET_CLASSES:
        continue

    # Load landmark sequence
    data = np.load(npy_path)
    length = data.shape[0]  # number of frames in the sequence

    # Store it
    class_to_lengths[gesture].append(length)


# -----------------------------------
# PLOTTING
# -----------------------------------
num_classes = len(TARGET_CLASSES)
rows = 4
cols = 3
plt.figure(figsize=(16, 18))

for idx, gesture in enumerate(sorted(TARGET_CLASSES), start=1):
    lengths = class_to_lengths[gesture]

    plt.subplot(rows, cols, idx)
    plt.hist(lengths, bins=20, color="steelblue", edgecolor="black")
    plt.title(gesture)
    plt.xlabel("Frames per Video")
    plt.ylabel("Count")

plt.tight_layout()
plt.show()
