import numpy as np
from pathlib import Path

INPUT_DIR = Path("datasets/jester/processed/train")
OUTPUT_DIR = Path("datasets/jester/processed_15frames/train")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FRAMES = 15


def interpolate_frame(a, b):
    #Return midpoint vector between frames a and b
    return (a + b) / 2.0


def normalize_sequence(seq, target_len=TARGET_FRAMES):
    seq = [frame for frame in seq]

    while len(seq) < target_len:
        #Compute diffs between adjacent frames
        diffs = [np.linalg.norm(seq[i+1] - seq[i]) for i in range(len(seq) - 1)]

        #Find largest difference
        index = int(np.argmax(diffs))

        #Interpolate midpoint between frames
        new_frame = interpolate_frame(seq[index], seq[index + 1])

        #Insert new frame AFTER idx
        seq.insert(index + 1, new_frame)

    while len(seq) > target_len:
        #Compute diffs between adjacent frames
        diffs = [np.linalg.norm(seq[i+1] - seq[i]) for i in range(len(seq) - 1)]

        #Find smallest difference
        index = int(np.argmin(diffs))

        #Delete the second frame in the pair
        del seq[index + 1]
        
    return np.array(seq)


def main():
    npy_files = list(INPUT_DIR.glob("*.npy"))
    print(f"Found {len(npy_files)} .npy files to normalize.")

    for npy_path in npy_files:
        seq = np.load(npy_path)
        seq_15 = normalize_sequence(seq, TARGET_FRAMES)

        out_path = OUTPUT_DIR / npy_path.name
        np.save(out_path, seq_15)

    print("All files normalized to 15 frames.")


if __name__ == "__main__":
    main()
