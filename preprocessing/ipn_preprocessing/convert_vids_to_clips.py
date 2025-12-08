import os
import cv2
import pandas as pd
from tqdm import tqdm

ORIG_DIR = "../../datasets/IPN/original_videos"
CLIP_DIR = "../../datasets/IPN/clip_frames"
CLASSES_CSV = "../../datasets/IPN/classIdx.txt"
ANNOT_CSV = "../../datasets/IPN/Annot_List.txt"

#Load Classes
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
        return

    if cap is not None:
        cap.release()

    video_path = os.path.join(ORIG_DIR, f"{video_file}.avi")
    cap = cv2.VideoCapture(video_path)
    current_video = video_file
    current_frame = 0


def fast_forward_to(frame_idx):
    #Decode frames until we reach the start index
    global current_frame, cap

    while current_frame < frame_idx:
        ret, _ = cap.read()
        if not ret:
            break
        current_frame += 1


def save_clip_frames(video_file, class_label, clip_name, start, end):
    global cap, current_frame

    clip_dir = os.path.join(CLIP_DIR, clip_name)

    if os.path.exists(clip_dir):
        #print(f"Skipping existing clip: {clip_name}")
        return

    open_video(video_file)

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

    #Sometimes indexes are not 0 base so we have to convert here to make sure
    start, end = row.t_start - 1, row.t_end - 1
    
    clip_name = f"{video}_{row.label}_{start}_{end}"

    save_clip_frames(video, class_label, clip_name, start, end)
    
if cap is not None:
    cap.release()

print("Done.")
