import os

# ---- must be before importing mediapipe ----
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 3 = ERROR only
os.environ["GLOG_minloglevel"] = "3"       # 3 = ERROR only
os.environ["MEDIAPIPE_LOG_LEVEL"] = "2"    # 2 = ERROR
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"  # <-- required to avoid GPU in subprocesses on macOS

import sys
import csv
import cv2
import time
import numpy as np
from pathlib import Path
import mediapipe as mp
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from utils import normalize

# ----------------------
# CONFIG
# ----------------------
MIN_FRAMES = 8

BASE_DATA_DIR = Path("datasets/jester/20bn-jester-v1")
PROCESSED_DIR = Path("datasets/jester/processed")

LABELS_TRAIN = Path("datasets/jester/jester_labels/jester_train_labels.csv")
LABELS_VALID = Path("datasets/jester/jester_labels/jester_validation_labels.csv")

mp_hands = mp.solutions.hands


def _suppress_cpp_logs():
    sys.stderr.flush()
    devnull = open(os.devnull, "w")
    os.dup2(devnull.fileno(), 2)


# ----------------------
# LANDMARK EXTRACTION
# ----------------------
def extract_landmarks(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    features = np.array([
        coord
        for lm in results.multi_hand_landmarks[0].landmark
        for coord in (lm.x, lm.y, lm.z)
    ])

    return normalize(features)


# ----------------------
# PROCESS SINGLE VIDEO
# ----------------------
def preprocess_vid(args):
    video_dir, split = args
    video_dir = Path(video_dir)
    video_id = video_dir.name

    frames = sorted(video_dir.glob("*.jpg"))
    if len(frames) < MIN_FRAMES:
        return

    _suppress_cpp_logs()

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1,
    ) as hands:

        sequence = []

        for frame_path in frames:
            image = cv2.imread(str(frame_path))
            features = extract_landmarks(image, hands)
            if features is not None:
                sequence.append(features)

        if len(sequence) < MIN_FRAMES:
            return

        out_path = PROCESSED_DIR / split / f"{video_id}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, np.array(sequence))


# ----------------------
# MULTIPROCESSING DRIVER
# ----------------------
def preprocess_data_parallel(split: str, labels_csv: Path, limit=None):

    # 1. Load labeled video IDs
    labeled_ids = []
    with open(labels_csv, "r") as f:
        reader = csv.reader(f, delimiter=";")
        for idx, (video_num, label) in enumerate(reader):
            if limit and idx >= limit:
                break
            labeled_ids.append(video_num)

    print(f"Found {len(labeled_ids)} labeled videos in: {labels_csv}")

    # 2. Find already processed videos
    processed_dir = PROCESSED_DIR / split
    processed_dir.mkdir(parents=True, exist_ok=True)
    existing_files = {p.stem for p in processed_dir.glob("*.npy")}
    print(f"Found {len(existing_files)} already processed videos.")

    # 3. Compute missing IDs
    missing_ids = [vid for vid in labeled_ids if vid not in existing_files]
    print(f"{len(missing_ids)} videos still need processing.")

    if not missing_ids:
        print("Nothing to do. All videos processed.")
        return

    # 4. Build paths list
    video_dirs = [BASE_DATA_DIR / vid for vid in missing_ids if (BASE_DATA_DIR / vid).exists()]
    print(f"Processing {len(video_dirs)} videos...\n")

    # 5. Multiprocessing setup
    n_workers = max(cpu_count() - 1, 1)
    print(f"Using {n_workers} worker processes.")

    args = [(str(v), split) for v in video_dirs]

    # 6. Run multiprocessing with progress bar
    with Pool(n_workers) as pool:
        for _ in tqdm(pool.imap_unordered(preprocess_vid, args), total=len(args)):
            pass


# ----------------------
# MAIN
# ----------------------
if __name__ == "__main__":
    start = time.time()

    preprocess_data_parallel(
        split="train",
        labels_csv=LABELS_TRAIN,
        limit=None,
    )

    print(f"\nTime taken: {time.time() - start:.2f} seconds")
