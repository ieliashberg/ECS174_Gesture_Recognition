# static_browser_control.py
'''
This is just a silly little program that uses the static gesture detection model to attempt to navigate a browser.
to start it, run the program and open your web browser, throw up a peace sign and it should begin detecting gestures.
Hold a gesture for one second and it will execute the corresponding command
'''

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import pyautogui
import mediapipe as mp

from static_gesture_net import StaticGestureNet
from utils import normalize

NET_PATH = "trained_models/static_gesture_net_v3.pt" #remember to change the \
CAM_INDEX = 0
CONF_MIN = 0.85 #sometimes the model will switch to a different gesture for a frame or so. So each frame if it meets this threshold we send it to the vote deque
SMOOTH_K = 7 #This number is the required majority vote for a gesture to be to be registered (in order to always have a majority keep this number odd)
DRAW_LANDMARKS = True

HOLD_SEC      = 2.0
REPEAT_EVERY  = 1.0
REPEATABLE = {"like", "dislike", "one", "peace", "three",}

KEYBINDINGS = {
    "like": lambda: pyautogui.scroll(+600),
    "dislike": lambda: pyautogui.scroll(-600),
    "one": lambda: pyautogui.press("tab"),
    "peace": lambda: pyautogui.hotkey("shift", "tab"),
    "three": lambda: pyautogui.hotkey("ctrl", "tab"),
    "fist": lambda: pyautogui.hotkey("ctrl", "c"),
    "palm": lambda: pyautogui.hotkey("ctrl", "v"),
    "stop": lambda: pyautogui.press("space"),
    "ok": lambda: pyautogui.press("enter"),
    "rock": lambda: pyautogui.hotkey("ctrl", "r"),
    "mute": lambda: pyautogui.hotkey("alt", "left"),
    "call": lambda: pyautogui.hotkey("alt", "right"),
}

#enable the key bindings
ENABLE_LABEL  = "two_up"
DISABLE_LABEL = "two_up_inverted"

#Helper Functions:
def extractFeatures(hand_landmarks) -> np.ndarray:
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])

    arr = np.asarray(row, dtype=np.float32).reshape(1, -1)
    
    arr = normalize(arr).astype(np.float32)
    
    return arr.reshape(-1)

def majorityVote(labels_deque):
    """
    Majority vote system: every frame it checks if a gesture's softmax magnitude is greater than CONF_MIN
    If it is, we add it to the labels deque and select the gesture with the most votes to trigger the keybind
    """
    if not labels_deque:
        return None
    counts = {}
    for lab in reversed(labels_deque):
        counts[lab] = counts.get(lab, 0) + 1
    return max(counts, key=counts.get)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loadNet = torch.load(NET_PATH, map_location=device)
    net = StaticGestureNet(loadNet["input_dim"], loadNet["num_classes"])
    net.load_state_dict(loadNet["model_state_dict"])
    net.to(device).eval()

    classes = loadNet["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
    mp_hands   = mp.solutions.hands

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    control_enabled = False
    recent_preds = deque(maxlen=SMOOTH_K)

    hold_label = None
    hold_start = None
    next_fire_at = None

    prev_t = time.perf_counter()

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5,
        max_num_hands=1
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            pred_label, pred_conf = None, 0.0

            if res.multi_hand_landmarks:
                if DRAW_LANDMARKS:
                    mp_drawing.draw_landmarks(
                        bgr, res.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                feat = extractFeatures(res.multi_hand_landmarks[0])
                x = torch.from_numpy(feat).unsqueeze(0).float().to(device)

                with torch.no_grad():
                    logits = net(x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                idx = int(np.argmax(probs))
                pred_label = classes[idx]
                pred_conf  = float(probs[idx])

                #prediction smoothing
                if pred_conf >= CONF_MIN:
                    recent_preds.append(pred_label)
                else:
                    recent_preds.append("unknown")

                smoothed = majorityVote(recent_preds)

                #toggles enabling/disabling
                if smoothed == ENABLE_LABEL and pred_conf >= CONF_MIN:
                    control_enabled = True
                    hold_label = hold_start = next_fire_at = None
                elif smoothed == DISABLE_LABEL and pred_conf >= CONF_MIN:
                    control_enabled = False
                    hold_label = hold_start = next_fire_at = None

                if control_enabled and smoothed not in (None, "unknown", ENABLE_LABEL, DISABLE_LABEL):
                    now = time.perf_counter()
                    if hold_label != smoothed:
                        hold_label = smoothed
                        hold_start = now
                        next_fire_at = hold_start + HOLD_SEC
                    else:
                        #handles firing once and repeating if repeat is enabled
                        if next_fire_at is not None and now >= next_fire_at:
                            if hold_label in KEYBINDINGS:
                                try:
                                    if hold_label in REPEATABLE:
                                        while now >= next_fire_at:
                                            KEYBINDINGS[hold_label]()
                                            next_fire_at += REPEAT_EVERY
                                    else:
                                        KEYBINDINGS[hold_label]()
                                        next_fire_at = float("inf")
                                except Exception as e:
                                    print(f"Action error for {hold_label}: {e}")
                else:
                    #this resets the hold state
                    hold_label = hold_start = next_fire_at = None
            else:
                recent_preds.append("unknown")
                hold_label = hold_start = next_fire_at = None

            # drawing displays and progress bar stuff
            disp = cv2.flip(bgr, 1)
            now = time.perf_counter()
            fps = 1.0 / (now - prev_t) if now > prev_t else 0.0
            prev_t = now

            status = "ENABLED" if control_enabled else "DISABLED"
            cv2.putText(disp, f"{fps:.1f} FPS | Control: {status}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 180, 255) if control_enabled else (0, 0, 255), 2)

            if pred_label:
                cv2.putText(disp, f"Pred: {pred_label} ({pred_conf:.2f})", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if hold_label and hold_start is not None and next_fire_at not in (None, float("inf")):
                base = HOLD_SEC if (time.perf_counter() < hold_start + HOLD_SEC) else REPEAT_EVERY
                interval_start = next_fire_at - base
                prog = np.clip((time.perf_counter() - interval_start) / base, 0.0, 1.0)
                x0, y0 = 10, 80
                x1 = 10 + int(200 * prog)
                cv2.rectangle(disp, (10, 80), (210, 100), (60, 60, 60), 2)
                cv2.rectangle(disp, (x0, y0), (x1, 100), (0, 220, 0), -1)
                cv2.putText(disp, f"Holding: {hold_label}", (220, 98),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

            cv2.imshow("Static gesture → browser", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #pyautogui has a safety feature where flinging the mouse to the corner will abort any keypressing
    pyautogui.FAILSAFE = True
    main()
