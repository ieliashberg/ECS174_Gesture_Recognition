import csv
import os
import mediapipe as mp
import numpy as np
from utils import normalize
import cv2
import time
mp_hands = mp.solutions.hands


MIN_FRAMES = 1
def extract_landmarks(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)  # process the image

    if results.multi_hand_landmarks:
        curr_feature_vec = []
        for ndx, lm in enumerate(results.multi_hand_landmarks[0].landmark):
            curr_feature_vec.extend([lm.x, lm.y, lm.z])
        features = np.array(curr_feature_vec)
        curr_features = normalize(features)
        return curr_features
    return None

def preprocess_vid(video_dir, train_or_validation, hands):
    sequence = []
    frames = sorted([f for f in os.listdir(video_dir) if f.endswith(".jpg")])
    for frame in frames:
        full_path = video_dir + "/" + frame
        image = cv2.imread(full_path)
        landmarks = extract_landmarks(image, hands)
        if landmarks is not None:
            sequence.append(landmarks)
    video_id = video_dir.split("/")[-1]
    output_dir = f"datasets/jester/processed/{train_or_validation}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{video_id}.npy"

    if len(sequence) == 0 or len(sequence) < MIN_FRAMES:
        return

    np.save(output_path, np.array(sequence))


# preprocess train
def preprocess_data(train_or_validation, labels_csv, hands):
    with open(labels_csv, "r") as f:
        reader = csv.reader(f, delimiter=";")
        count = 0
        for row in reader:
            if count > 100:
                break
            video_num = row[0]
            label = row[1]
            preprocess_vid(f"datasets/jester/20bn-jester-v1/{video_num}", train_or_validation, hands)
            count += 1

with mp_hands.Hands( model_complexity=0, min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands = 1) as hands:
    # preprocess_data("train", "datasets/jester/jester_labels/jester_train_labels.csv", hands)
    start = time.time()
    preprocess_data("validation", "datasets/jester/jester_labels/jester_validation_labels.csv", hands)
    end = time.time()
    print(f"time taken = {end - start}")

# video_dirs = os.listdir("datasets/jester/20bn-jester-v1")
# my_dict = {}
# for video in video_dirs:
#     if video.isdigit():
#         if len(os.listdir(f"datasets/jester/20bn-jester-v1/{video}")) in my_dict:
#             my_dict[len(os.listdir(f"datasets/jester/20bn-jester-v1/{video}"))] += 1
#         else:
#             my_dict[len(os.listdir(f"datasets/jester/20bn-jester-v1/{video}"))] = 1
# print(sorted(my_dict.items()))

video_dirs = os.listdir("datasets/jester/processed/validation")
print(len(video_dirs))