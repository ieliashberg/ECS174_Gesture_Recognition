from pathlib import Path
import mediapipe as mp
import cv2
import time
import random
from collections import defaultdict
from tqdm import tqdm


NUM_PER_CLASS = 20

ROOT = Path("../datasets/IPN/clip_frames")
OUT = Path("../datasets/IPN/sample_overlays")

SEED = 11

def extract_gesture(clip_name: str) -> str:
    return clip_name.split("_")[-3]

def findLandmarks(clip_dir: Path, out_root: Path, hands, gesture: str):

    out_dir = out_root / gesture / clip_dir.name

    # if out_dir.exists():
    #     print(f"[SKIP] {clip_dir.name} already processed")
    #     return

    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(clip_dir.glob("*.jpg"))
    for frame in frames:
        bgr = cv2.imread(str(frame))
        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

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

if __name__ == "__main__":
    if SEED is not None:
        random.seed(SEED)

    OUT.mkdir(parents=True, exist_ok=True)
    gesture_to_clips = defaultdict(list)

    for clip_dir in ROOT.iterdir():
        if clip_dir.is_dir():
            gesture = extract_gesture(clip_dir.name)
            gesture_to_clips[gesture].append(clip_dir.name)

    sampled = {}
    print("Sampling clips per gesture class...\n")

    for gesture, clips in gesture_to_clips.items():
        k = min(NUM_PER_CLASS, len(clips))
        sampled_clips = random.sample(clips, k)
        sampled[gesture] = sampled_clips
        print(f"{gesture}: {k} clips sampled")

    total = sum(len(v) for v in sampled.values())
    print(f"\nTotal sampled clips = {total}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    start = time.time()
    print("\nProcessing clips...\n")
    worklist = [(gesture, clip_name)
                for gesture, clips in sampled.items()
                for clip_name in clips]

    for gesture, clip_name in tqdm(worklist, desc="Overlay clips", ncols=80):
        clip_path = ROOT / clip_name
        if clip_path.exists():
            findLandmarks(clip_path, OUT, hands, gesture)
        else:
            print(f"[WARN] Missing clip directory: {clip_name}")

    hands.close()

    end = time.time()
    print(f"\nProcessing complete. Time taken = {end - start:.2f} seconds\n")
