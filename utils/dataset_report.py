"""
This program needs to:
Count gestures per class, 
Count sequences under MIN_FRAMES per class,
Compute sequence length statistics,
Plot histogram of overall lengths
"""

from pathlib import Path
import numpy as np
import statistics
import matplotlib.pyplot as plt
from collections import defaultdict

DATASET_DIR = Path("../datasets/IPN/IPN_dynamic_npz_normalized")
MIN_FRAMES = 10

def extract_label(filename: str) -> str:
    """Label is the 3rd-from-last token in IPN filenames"""
    return filename.split("_")[-3]


def load_lengths_by_label():
    lengths = defaultdict(list)
    short_counts = defaultdict(int)

    npz_files = sorted(DATASET_DIR.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {DATASET_DIR}")

    for f in npz_files:
        with np.load(f) as npz:

            if "x" in npz:
                seq = npz["x"]
            elif "data" in npz:
                seq = npz["data"][:, :63]
            else:
                raise KeyError(f"{f} missing expected key 'x' or 'data'")

        label = extract_label(f.stem)
        T = seq.shape[0]
        lengths[label].append(T)

        if T < MIN_FRAMES:
            short_counts[label] += 1

    return lengths, short_counts


def print_stats(lengths, short_counts):
    all_lengths = [l for v in lengths.values() for l in v]

    print("\n=== IPN GESTURE DATASET REPORT ===")
    print(f"Dataset dir        : {DATASET_DIR}")
    print(f"Total gesture clips: {sum(len(v) for v in lengths.values())}")
    print(f"Number of classes  : {len(lengths)}")
    print(f"Min length         : {min(all_lengths)}")
    print(f"Max length         : {max(all_lengths)}")
    print(f"Mean length        : {statistics.mean(all_lengths):.2f}")
    print(f"Median length      : {statistics.median(all_lengths):.2f}")
    print(f"Std dev            : {statistics.pstdev(all_lengths):.2f}")
    print("------------------------------------------------------------")

    print("\nPer-Class Statistics:")
    print("Class | Count | <10 frames | Min | Max | Mean | Median")
    for label, lens in sorted(lengths.items()):
        print(
            f"{label:>5} | {len(lens):5d} | {short_counts[label]:10d} | "
            f"{min(lens):3d} | {max(lens):3d} | "
            f"{statistics.mean(lens):6.2f} | {statistics.median(lens):6.2f}"
        )

def plot_histogram(lengths):
    all_lengths = [l for v in lengths.values() for l in v]

    plt.figure(figsize=(10, 6))
    plt.hist(all_lengths, bins=40, edgecolor="black")
    plt.title("Sequence Length Distribution — All Gestures")
    plt.xlabel("Frames")
    plt.ylabel("Count")
    plt.grid(alpha=0.35)
    plt.show()

if __name__ == "__main__":
    lengths, short_counts = load_lengths_by_label()
    print_stats(lengths, short_counts)
    plot_histogram(lengths)
