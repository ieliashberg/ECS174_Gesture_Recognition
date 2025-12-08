import os
import mediapipe as mp
import cv2
import numpy as np
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count

ROOT = "../../datasets/IPN/clip_frames"
OUT  = "../../datasets/IPN/IPN_dynamic_npz"


def process_frame(img, hands):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    vec = np.zeros(64, dtype=np.float32)

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        coords = []
        for p in lm.landmark:
            coords.extend([p.x, p.y, p.z])
        vec[:63] = coords
        vec[63] = 1.0

    return vec


def process_clip(clip_dir_path: str):
    """Worker entry point: create MediaPipe Hands locally per process"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    clip_name = os.path.basename(clip_dir_path)

    #Output directory for all .npz files
    out_file = os.path.join(OUT, f"{clip_name}.npz")
    if os.path.exists(out_file):
        hands.close()
        return

    frame_files = sorted(
        f for f in os.listdir(clip_dir_path)
        if f.lower().endswith(".jpg")
    )

    if not frame_files:
        hands.close()
        return

    seq = []
    for fname in frame_files:
        img_path = os.path.join(clip_dir_path, fname)
        img = cv2.imread(img_path)
        if img is not None:
            seq.append(process_frame(img, hands))

    if seq:
        seq = np.stack(seq).astype(np.float32)
        np.savez_compressed(out_file, data=seq)

    hands.close()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)

    clips = [
        os.path.join(ROOT, d)
        for d in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, d))
    ]

    print(f"Total clips to process: {len(clips)}\n")

    start = time.time()

    #HAndling multi-processing
    num_workers = max(cpu_count() - 1, 1)
    print(f"Using {num_workers} workers")

    with Pool(num_workers) as p:
        for _ in tqdm(
            p.imap_unordered(process_clip, clips),
            total=len(clips),
            desc="NPZ conversion",
            ncols=80
        ):
            pass

    print(f"\nDone. Time taken = {time.time() - start:.2f} seconds")
