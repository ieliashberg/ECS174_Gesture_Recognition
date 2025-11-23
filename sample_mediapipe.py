from pathlib import Path
import mediapipe as mp
import numpy as np
from utils import normalize
import cv2
import time
import random

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

NUM_VIDS = 5
ROOT = r"..\JesterDataset\20bn-jester-v1"
OUT = r"datasets\mpOverlays"

SEED = 11
"""
This file randomly samples NUM_VIDS videos from the jester dataset and sends them through MediaPipe
The output can be verified in datasets/mOverlays
"""

def findLandmarks(frame_path: Path, out_root: Path):
    out_dir = out_root / frame_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    hands = mp_hands.Hands(
        static_image_mode = False,
        max_num_hands = 1,
        model_complexity=0, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    try:
        frames = sorted(frame_path.glob("*.jpg"))
        
        for frame in frames:
            bgr = cv2.imread(str(frame))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(bgr, lm, mp_hands.HAND_CONNECTIONS)

            cv2.imwrite(str(out_dir / frame.name), bgr)
    finally:
        hands.close()

root = Path(ROOT)
out_root = Path(OUT)
out_root.mkdir(parents=True, exist_ok=True)

vid_dirs = [p for p in root.iterdir() if p.is_dir()]
if not vid_dirs:
    raise SystemExit(f"No video folders found under {root}")

if SEED is not None:
    random.seed(SEED)

sample = random.sample(vid_dirs, k=min(NUM_VIDS, len(vid_dirs)))

start = time.time()
for vid in sample:
    findLandmarks(vid, out_root)

end = time.time()
print(f"time taken = {end - start}")