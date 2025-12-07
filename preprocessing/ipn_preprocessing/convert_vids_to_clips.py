import os
import cv2
import pandas as pd
from tqdm import tqdm

ORIG_DIR = "../../datasets/IPN/original_videos"
CLIP_DIR = "../../datasets/IPN/clip_frames"  # main directory for all clip subdirs
CLASSES_CSV = "../../datasets/IPN/classIdx.txt"
ANNOT_CSV = "../../datasets/IPN/Annot_List.txt"

# Load class labels
class_df = pd.read_csv(CLASSES_CSV)
id_to_label = dict(zip(class_df.id, class_df.label))

annot_df = pd.read_csv(ANNOT_CSV, sep=",")

# Ensure ordering by video + frame start
annot_df = annot_df.sort_values(by=["video", "t_start"])

os.makedirs(CLIP_DIR, exist_ok=True)

# Decoder reuse state
current_video = None
cap = None
current_frame = 0


def open_video(video_file):
    """Open this video if not already open."""
    global current_video, cap, current_frame

    if current_video == video_file:
        return  # already opened

    if cap is not None:
        cap.release()

    video_path = os.path.join(ORIG_DIR, f"{video_file}.avi")
    cap = cv2.VideoCapture(video_path)
    current_video = video_file
    current_frame = 0


def fast_forward_to(frame_idx):
    """Decode frames until we reach the requested start."""
    global current_frame, cap

    while current_frame < frame_idx:
        ret, _ = cap.read()
        if not ret:
            break
        current_frame += 1


def save_clip_frames(video_file, class_label, clip_name, start, end):
    """
    Each clip gets its own directory:
    clip_frames/<clip_name>/frame_00000.jpg
    """
    global cap, current_frame

    clip_dir = os.path.join(CLIP_DIR, clip_name)

    # -----------------------------
    # NEW: SKIP IF CLIP ALREADY EXISTS
    # -----------------------------
    if os.path.exists(clip_dir):
        # print(f"Skipping existing clip: {clip_name}")  # optional
        return

    # Only open video once unless switching files
    open_video(video_file)

    # If we already passed this start, reset the video
    if current_frame > start:
        cap.release()
        open_video(video_file)

    fast_forward_to(start)

    os.makedirs(clip_dir, exist_ok=True)

    saved = 0
    while current_frame <= end:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(clip_dir, f"frame_{saved:05d}.jpg")
        cv2.imwrite(frame_path, frame)

        saved += 1
        current_frame += 1


print("Extracting clip frames...")
for _, row in tqdm(annot_df.iterrows(), total=len(annot_df), desc="Clips processed"):
    video = row.video
    class_label = id_to_label[row.id]

    # Convert to 0-based frame indexing
    start, end = row.t_start - 1, row.t_end - 1

    # Clip directory will uniquely identify video + gesture + range
    clip_name = f"{video}_{row.label}_{start}_{end}"

    save_clip_frames(video, class_label, clip_name, start, end)

# Cleanup last video
if cap is not None:
    cap.release()

print("Done.")
