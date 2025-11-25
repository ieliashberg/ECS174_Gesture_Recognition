import numpy as np
from pathlib import Path

INPUT_DIR = Path("datasets/jester/processed/train")
OUTPUT_DIR = Path("datasets/jester/processed_15frames/train")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FRAMES = 15


def interpolate_frame(a, b):
    """Return midpoint vector between frames a and b."""
    return (a + b) / 2.0


def normalize_sequence(seq, target_len=TARGET_FRAMES):
    """
    Normalize seq (list or np.ndarray of frames) to exactly target_len.
    Frames are 63D landmark vectors.
    """
    seq = [frame for frame in seq]  # ensure list of arrays

    # ---------------------------
    # Case 1: Sequence too short
    # ---------------------------
    while len(seq) < target_len:
        # Compute diffs between adjacent frames
        diffs = [np.linalg.norm(seq[i+1] - seq[i]) for i in range(len(seq) - 1)]

        # Find largest difference
        idx = int(np.argmax(diffs))

        # Interpolate midpoint between frames idx and idx+1
        new_frame = interpolate_frame(seq[idx], seq[idx + 1])

        # Insert new frame AFTER idx
        seq.insert(idx + 1, new_frame)

    # ---------------------------
    # Case 2: Sequence too long
    # ---------------------------
    while len(seq) > target_len:
        # Compute diffs between adjacent frames
        diffs = [np.linalg.norm(seq[i+1] - seq[i]) for i in range(len(seq) - 1)]

        # Find smallest difference — frames are redundant here
        idx = int(np.argmin(diffs))

        # Delete the second frame in the pair (idx + 1)
        del seq[idx + 1]

    # Return final sequence as np array
    return np.array(seq)


def main():
    npy_files = list(INPUT_DIR.glob("*.npy"))
    print(f"Found {len(npy_files)} .npy files to normalize.")

    for npy_path in npy_files:
        seq = np.load(npy_path)        # shape (T, 63)
        seq_15 = normalize_sequence(seq, TARGET_FRAMES)

        out_path = OUTPUT_DIR / npy_path.name
        np.save(out_path, seq_15)

    print("All files normalized to 15 frames.")


if __name__ == "__main__":
    main()
