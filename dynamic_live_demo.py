"""
Analyze IPN_dynamic dataset with human-readable gesture names:
  - Per-class length statistics (mapped to actual gesture names)
  - Overall dataset stats
  - Histogram of sequence lengths
"""

from pathlib import Path
import numpy as np
import statistics
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------------------------------------------------------- #
# CONFIG
# ---------------------------------------------------------------------------- #

DST = Path("datasets/IPN/IPN_dynamic_normalized")

GESTURE_NAMES = {
    "G01": "Click with one finger",
    "G02": "Click with two fingers",
    "D0X": "no_gesture",
    "B0A": "Pointing with one finger",
    "B0B": "Pointing with two fingers",
    "G03": "Throw up",
    "G04": "Throw down",
    "G05": "Throw left",
    "G06": "Throw right",
    "G07": "Open twice",
    "G08": "Double click with one finger",
    "G09": "Double click with two fingers",
    "G10": "Zoom in",
    "G11": "Zoom out",
}

# ---------------------------------------------------------------------------- #
# LOAD DATA
# ---------------------------------------------------------------------------- #

def load_lengths_by_label():
    files = sorted(DST.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {DST}")

    lengths_by_label = defaultdict(list)

    for f in files:
        with np.load(f) as npz:
            label = str(npz["label"])
            x = npz["x"]
            lengths_by_label[label].append(x.shape[0])

    return lengths_by_label


# ---------------------------------------------------------------------------- #
# PRINT STATISTICS WITH HUMAN LABELS
# ---------------------------------------------------------------------------- #

def print_stats(lengths_by_label):
    all_lengths = [l for seq in lengths_by_label.values() for l in seq]

    print("\n================= DATASET REPORT =================")
    print(f"Total gestures        : {sum(len(v) for v in lengths_by_label.values())}")
    print(f"Number of classes     : {len(lengths_by_label)}")
    print(f"Min sequence length   : {min(all_lengths)}")
    print(f"Max sequence length   : {max(all_lengths)}")
    print(f"Mean sequence length  : {statistics.mean(all_lengths):.2f}")
    print(f"Median sequence length: {statistics.median(all_lengths):.2f}")
    print(f"Std deviation         : {statistics.pstdev(all_lengths):.2f}")
    print("---------------------------------------------------")
    print("Per-Class Sequence Length Statistics:\n")

    for code, seq_lengths in sorted(lengths_by_label.items()):
        name = GESTURE_NAMES.get(code, code)
        print(
            f"{code:>3} | {name:35s} | "
            f"count={len(seq_lengths):4d} | "
            f"min={min(seq_lengths):4d} | "
            f"max={max(seq_lengths):4d} | "
            f"mean={statistics.mean(seq_lengths):7.2f} | "
            f"median={statistics.median(seq_lengths):7.2f}"
        )

    print("===================================================\n")


# ---------------------------------------------------------------------------- #
# HISTOGRAM OF ALL LENGTHS
# ---------------------------------------------------------------------------- #

def plot_histogram(lengths_by_label):
    all_lengths = [l for seq in lengths_by_label.values() for l in seq]

    plt.figure(figsize=(10, 6))
    plt.hist(all_lengths, bins=30, edgecolor='black')
    plt.title("Sequence Length Distribution — All Gestures")
    plt.xlabel("Frames per Gesture")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.4)
    plt.show()


# ---------------------------------------------------------------------------- #
# MAIN
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    lengths_by_label = load_lengths_by_label()
    print_stats(lengths_by_label)
    plot_histogram(lengths_by_label)
