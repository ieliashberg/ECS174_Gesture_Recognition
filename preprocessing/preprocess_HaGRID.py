from pathlib import Path
import json, csv
import cv2
import mediapipe as mp
import time

#This program will loop through the haGRID root directory:
#   Extract hand landmarks from the images
#   Store the landmarks in a csv based on their label
#some images won't be able to find hand landmarks, I can just return none and store no image

#Root directories:
IMG_ROOT = Path("..\HaGRID\hagrid-sample-30k-384p\hagrid_30k")
ANN_ROOT = Path("..\HaGRID\hagrid-sample-30k-384p\hagrid_ann_train_val")
OUT_CSV = Path("datasets\hagrid_static_dataset.csv")

#Search through annotations for user ID of each image along with label
uuid_to_labels: dict[str, list] = {}

def _add_aliases(key: str, labels: list):
    # Store multiple keys so lookups succeed regardless of how the JSON keyed it
    uuid_to_labels[key] = labels
    uuid_to_labels[key.lower()] = labels
    stem = Path(key).stem
    uuid_to_labels[stem] = labels
    uuid_to_labels[stem.lower()] = labels

for j in sorted(ANN_ROOT.glob("*.json")):
    with j.open("r", encoding="utf-8") as f:
        data = json.load(f)  # {uuid: {"labels":[...], ...}}
    for key, rec in data.items():
        _add_aliases(key, rec.get("labels", []))

#finds the first gesture label
def choose_label(labels: list[str]) -> str | None:
    for label in labels:
        if label != "no_gesture":
            return label
    return "no_gesture"

#create CSV
header = [f"{c}{i}" for i in range(21) for c in ("x","y","z")] + ["label"]
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
new_file = not OUT_CSV.exists()
fout = OUT_CSV.open("a", newline="", encoding="utf-8")
writer = csv.writer(fout)
if new_file:
    writer.writerow(header)

#Create Hands instance
hands = mp.solutions.hands.Hands(
    static_image_mode=True, 
    max_num_hands=1,
    model_complexity=0, 
    min_detection_confidence=0.5
)

image_files = list(IMG_ROOT.rglob("*"))
image_files = [p for p in image_files
               if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")]

print(f"Found {len(image_files)} image files under {IMG_ROOT}")

count = 0 
kept = 0
for images in image_files:
    #I checked and I think all images are either jpg or jpeg but cant be sure
    if images.suffix.lower() not in (".jpg", ".jpeg"):
        continue

    uuid = images.stem
    labels = uuid_to_labels.get(uuid)
    if not labels:
        continue

    label = choose_label(labels)
    bgr = cv2.imread(str(images), cv2.IMREAD_COLOR)
    if bgr is None:
        continue

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if not res.multi_hand_landmarks:
        continue

    row = []
    for lm in res.multi_hand_landmarks[0].landmark:
        row.extend([lm.x, lm.y, lm.z])
    row.append(label)
    writer.writerow(row)
    kept += 1

    count += 1
    if count % 1000 == 0:
        print(f"completed {count} images, wrote {kept} rows")

hands.close()
fout.close()
print(f"Done. wrote {kept} rows")