from pathlib import Path
import mediapipe as mp
import cv2
import time
import random
import csv
from collections import defaultdict

# -------------------------
# CONFIG
# -------------------------
NUM_PER_CLASS = 50

ROOT = Path("datasets/jester/20bn-jester-v1")
OUT = Path("datasets/overlays")

TRAIN_LABELS = Path("datasets/jester/jester_labels/jester_train_labels.csv")
ALL_LABELS = Path("datasets/jester/jester_labels/labels.csv")

SEED = 11


# -------------------------
# LOAD LABEL LIST
# -------------------------
def load_all_labels(path: Path):
    labels = []
    with open(path, "r") as f:
        for line in f:
            label = line.strip()
            if label:
                labels.append(label)
    return labels


# -------------------------
# LOAD VIDEO → CLASS MAPPING
# -------------------------
def load_video_to_class_map(csv_path: Path):
    mapping = defaultdict(list)  # gesture → list of video_ids
    with open(csv_path, "r") as f:
        reader = csv.reader(f, delimiter=";")
        for vid_str, gesture in reader:
            if vid_str.isdigit():
                mapping[gesture].append(vid_str)
    return mapping


# -------------------------
# PROCESS A SINGLE VIDEO DIRECTORY
# -------------------------
def findLandmarks(video_path: Path, out_root: Path, hands, gesture: str):
    out_dir = out_root / gesture / video_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(video_path.glob("*.jpg"))
    for frame in frames:
        bgr = cv2.imread(str(frame))
        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    bgr,
                    lm,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )

        cv2.imwrite(str(out_dir / frame.name), bgr)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    if SEED is not None:
        random.seed(SEED)

    OUT.mkdir(parents=True, exist_ok=True)

    # Load list of all gestures (labels.csv)
    all_gestures = load_all_labels(ALL_LABELS)

    # Load mapping gesture → video IDs
    gesture_to_videos = load_video_to_class_map(TRAIN_LABELS)

    # Sample 50 videos per gesture
    gesture_to_sampled = {}
    print("Sampling videos per gesture class...\n")

    for gesture in all_gestures:
        videos = gesture_to_videos.get(gesture, [])
        if not videos:
            print(f"[WARN] No videos found for: {gesture}")
            continue

        k = min(NUM_PER_CLASS, len(videos))
        sampled = random.sample(videos, k)
        gesture_to_sampled[gesture] = sampled

        print(f"{gesture}: {k} videos sampled")

    print("\nTotal sampled videos =",
          sum(len(v) for v in gesture_to_sampled.values()))

    # Create a single MediaPipe Hands instance
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    start = time.time()

    # Process each sampled video
    print("\nProcessing videos...\n")
    for gesture, vid_list in gesture_to_sampled.items():
        for vid_id in vid_list:
            video_path = ROOT / vid_id
            if video_path.exists() and video_path.is_dir():
                findLandmarks(video_path, OUT, hands, gesture)
            else:
                print(f"[WARN] Missing video directory: {vid_id} (class: {gesture})")

    hands.close()

    end = time.time()
    print(f"\nProcessing complete. Time taken = {end - start:.2f} seconds\n")
