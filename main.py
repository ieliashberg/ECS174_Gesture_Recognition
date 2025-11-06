import mediapipe as mp
import cv2
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)  # could be 2 or something else
prev_time = time.perf_counter()  # for fps

with mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.8, min_tracking_confidence=0.5
) as hands:
    while cap.isOpened:  # While capturin from the video
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )  # convert default from open cv (BGR) to RGB (which is what mediapipe needs)

        image.flags.writeable = False  # for numpy arrays, we don't want to be able to change the actual array (setting to False)
        results = hands.process(image)  # process the image
        image.flags.writeable = (
            True  # changing back to true if we want to annotate the image
        )

        image = cv2.cvtColor(
            image, cv2.COLOR_RGB2BGR
        )  # need to convert back to BGR default for open cv to display
        print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
        flipped_img = cv2.flip(image, 1)

        #putting fps on the screen
        curr_time = time.perf_counter()
        if curr_time > prev_time:
            fps = 1.0 / (curr_time - prev_time)
        else:
            fps = 0
        prev_time = curr_time
        cv2.putText(
            flipped_img,
            f"{fps:.1f} FPS",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            4,
        )

        cv2.imshow("Hand Tracking", flipped_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release
cv2.destroyAllWindows()
