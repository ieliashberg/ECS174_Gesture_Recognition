import mediapipe as mp
import cv2
import csv
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

#CSVs should have a structure like this: (x0,y0,z0,x1,y1,z1...x20,y20,z20,label)
#So we write this as the head of the csv file
csv_path = "static_dataset.csv"
if not os.path.exists(csv_path):
    header = [f"{coord}{i}" for i in range(21) for coord in ["x", "y", "z"]] + ["label"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

key_to_label = {
    ord('1'): "thumbs_up",
    ord('2'): "open_palm",
    ord('3'): "peace",
    ord('4'): "fist",
    ord('5'): "no_gesture"
}
with mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands = 1
) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        key = cv2.waitKey(1) & 0xFF
        image = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        image = cv2.cvtColor(
            image, cv2.COLOR_RGB2BGR
        )
        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
            if key in key_to_label:
                row = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    row.extend([lm.x, lm.y, lm.z])
                row.append(key_to_label[key])
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                print(f"Saved sample for {key_to_label[key]}")


        flipped_img = cv2.flip(image, 1)

        current_label = key_to_label.get(key, "")
        if current_label:
            cv2.putText(flipped_img, f"Label: {current_label}", (30, 50),
   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Hand Tracking", flipped_img)
        if key == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
