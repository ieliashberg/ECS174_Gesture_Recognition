import mediapipe as mp
import cv2
import csv
import os

#dictionary storing how many labels there are
label_to_count = {}

#count how many of each gesture there are in the dataset (want there to be all roughly the same amount)
with open("static_dataset.csv", "r", newline = "") as f:
    reader = csv.reader(f)
    label_to_count = {}
    for row in reader:
        label = row[-1]
        if label == "label":
            continue
        if row[-1] in label_to_count:
            label_to_count[label] += 1
        else:
            label_to_count[label] = 1
    print(label_to_count)

