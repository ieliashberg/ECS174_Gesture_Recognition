import csv
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("datasets/jester/processed")
OUTPUT_DIR = Path("datasets/jester/processed_npz")

LABELS_TXT = Path("datasets/jester/jester_labels/labels.csv")
TRAIN_LABELS_CSV = Path("datasets/jester/jester_labels/jester_train_labels.csv")
VALID_LABELS_CSV = Path("datasets/jester/jester_labels/jester_validation_labels.csv")


def load_all_labels(labels_txt: Path):
    """
    Load the master list of labels from labels.csv and build
    a mapping from gesture name -> integer class id.
    """
    labels = []
    with labels_txt.open("r") as f:
        for line in f:
            label = line.strip()
            if label:
                labels.append(label)

    gesture_to_id = {gesture: idx for idx, gesture in enumerate(labels)}
    return labels, gesture_to_id


def load_video_to_label_map(csv_path: Path):
    """
    Load mapping from video_id (string) -> gesture label (string)
    from a jester_*_labels.csv file.
    """
    mapping = {}
    with csv_path.open("r") as f:
        reader = csv.reader(f, delimiter=";")
        for video_id, gesture in reader:
            mapping[video_id] = gesture
    return mapping


def convert_split(split_name: str, video_to_label: dict, gesture_to_id: dict):
    """
    Convert .npy files in datasets/jester/processed/<split_name>/
    into .npz files in datasets/jester/processed_npz/<split_name>/,
    attaching label and label_id metadata.
    """
    input_dir = PROCESSED_DIR / split_name
    output_dir = OUTPUT_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_dir.glob("*.npy"))
    print(f"[{split_name}] Found {len(npy_files)} .npy files")

    converted = 0
    skipped_no_label = 0
    skipped_unknown_gesture = 0

    for npy_path in npy_files:
        video_id = npy_path.stem

        #Look up gesture name
        gesture = video_to_label.get(video_id)
        if gesture is None:
            skipped_no_label += 1
            print(f"[WARN] No label found for video_id={video_id} in split={split_name}, skipping.")
            continue

        #Look up label id
        label_id = gesture_to_id.get(gesture)
        if label_id is None:
            skipped_unknown_gesture += 1
            print(f"[WARN] Gesture '{gesture}' not in labels list for video_id={video_id}, skipping.")
            continue

        #Load sequence
        sequence = np.load(npy_path)
        sequence = sequence.astype(np.float32)

        #Save as npz with metadata
        out_path = output_dir / f"{video_id}.npz"
        np.savez(
            out_path,
            sequence=sequence,
            label=gesture,
            label_id=label_id,
            video_id=video_id,
        )
        converted += 1

    print(f"[{split_name}] Converted: {converted}")
    print(f"[{split_name}] Skipped (no label): {skipped_no_label}")
    print(f"[{split_name}] Skipped (unknown gesture): {skipped_unknown_gesture}")
    print()


def main():
    labels, gesture_to_id = load_all_labels(LABELS_TXT)
    print(f"Loaded {len(labels)} labels from {LABELS_TXT}")

    train_video_to_label = load_video_to_label_map(TRAIN_LABELS_CSV)
    valid_video_to_label = load_video_to_label_map(VALID_LABELS_CSV)

    convert_split("train", train_video_to_label, gesture_to_id)
    convert_split("validation", valid_video_to_label, gesture_to_id)


if __name__ == "__main__":
    main()
