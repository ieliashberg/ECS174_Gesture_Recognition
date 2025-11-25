import os

# ---- must be before importing mediapipe ----
# Reduce TensorFlow Lite / MediaPipe log levels
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 3 = ERROR only
os.environ["GLOG_minloglevel"] = "3"       # 3 = ERROR only (for glog-style logs)
os.environ["MEDIAPIPE_LOG_LEVEL"] = "2"    # 2 = ERROR

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
    """
    Hard-suppress C++ logs (glog/absl) by redirecting stderr to /dev/null
    for this process. Called inside each worker before creating Hands().
    """
    sys.stderr.flush()
    devnull = open(os.devnull, "w")
    os.dup2(devnull.fileno(), 2)  # dup2(fd, 2) => replace stderr at OS level


# ----------------------
# LANDMARK EXTRACTION
# ----------------------
def extract_landmarks(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    # Flatten (21 landmarks × 3 coords)
    features = np.array([
        coord
        for lm in results.multi_hand_landmarks[0].landmark
        for coord in (lm.x, lm.y, lm.z)
    ])

    return normalize(features)


# ----------------------
# PROCESS SINGLE VIDEO (Worker Function)
# ----------------------
def preprocess_vid(args):
    video_dir, split = args
    video_dir = Path(video_dir)
    video_id = video_dir.name

    frames = sorted(video_dir.glob("*.jpg"))
    if len(frames) < MIN_FRAMES:
        return  # skip short videos

    # Suppress MediaPipe C++ logs in this worker process
    _suppress_cpp_logs()

    # Each process creates its own Hands instance
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
            return  # skip empty or low-quality videos

        # Save output
        out_path = PROCESSED_DIR / split / f"{video_id}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, np.array(sequence))


# ----------------------
# MULTIPROCESSING DRIVER + TQDM
# ----------------------
def preprocess_data_parallel(split: str, labels_csv: Path, limit=None):
    # Build list of videos
    video_dirs = []
    with open(labels_csv, "r") as f:
        reader = csv.reader(f, delimiter=";")
        for idx, (video_num, label) in enumerate(reader):
            if limit and idx >= limit:
                break
            video_dir = BASE_DATA_DIR / video_num
            if video_dir.exists():
                video_dirs.append(video_dir)

    total = len(video_dirs)
    print(f"Loaded {total} videos for split='{split}'")

    # Number of worker processes
    n_workers = max(cpu_count() - 1, 1)
    print(f"Using {n_workers} worker processes\n")

    args = [(str(v), split) for v in video_dirs]

    # Multiprocessing with progress bar
    with Pool(n_workers) as pool:
        for _ in tqdm(pool.imap_unordered(preprocess_vid, args), total=total):
            pass


# ----------------------
# MAIN
# ----------------------
if __name__ == "__main__":
    start = time.time()

    preprocess_data_parallel(
        split="test",
        labels_csv=LABELS_TRAIN,
        limit=None,   # Set to None for full dataset
    )

    print(f"\nTime taken: {time.time() - start:.2f} seconds")
