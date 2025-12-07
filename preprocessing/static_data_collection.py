import mediapipe as mp
import cv2
import csv
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)  # could be 2 or something else

# write the header for the csv file (x0,y0,z0,x1,y1,z1...x20,y20,z20,label)
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
    while cap.isOpened():  # While capturin from the video
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        key = cv2.waitKey(1) & 0xFF
        image = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )  # convert default from open cv (BGR) to RGB (which is what mediapipe needs)

        image.flags.writeable = False  # for numpy arrays, we don't want to be able to change the actual array (setting to False) (speed up)
        results = hands.process(image)  # process the image
        image.flags.writeable = True  # changing back to true if we want to annotate the image

        image = cv2.cvtColor(
            image, cv2.COLOR_RGB2BGR
        )  # need to convert back to BGR default for open cv to display
        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
            # if the key pressed corresponds to a labeled gesture, add it to the dataset
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

        # write the label on the screen for visual feedback
        current_label = key_to_label.get(key, "")
        if current_label:
            cv2.putText(flipped_img, f"Label: {current_label}", (30, 50),
   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Hand Tracking", flipped_img)
        if key == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
