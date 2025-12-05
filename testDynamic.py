# dynamic_main.py
import cv2
import time
import numpy as np
import torch
from collections import deque
import mediapipe as mp

from dynamic_gesture_net import DynamicGestureNet
from utils import normalize

MODEL_PATH = "trained_models\dynamic_gesture_net_best_v7.pt"
CAM_INDEX = 0
WINDOW_SIZE = 30
STRIDE = 15
MIN_CONF = 0.70
DRAW_LANDMARKS = True
NORMALIZER = normalize #OR normalize

IPN_DECODE = {
    "G01": "Click with one finger",
    "G02": "Click with two fingers",
    "D0X": "no_gesture",
    "B0A": "Pointing with one finger",
    "B0B": "Pointing with two fingers",
    "G03": "Throw up",
    "G04": "Throw down",
    "G05": "Throw left",
    "G06": "Throw right",
    "G07": "Open twice",
    "G08": "Double click with one finger",
    "G09": "Double click with two fingers",
    "G10": "Zoom in",
    "G11": "Zoom out",
}

def decode_ipn(label: str) -> str:
    return IPN_DECODE.get(label, label)

def _normalize_shape(newVect: np.ndarray) -> np.ndarray:
    v = np.asarray(newVect, dtype=np.float32).reshape(1, -1)
    if NORMALIZER:
        out = normalize(v)
    else:
        out = v
    return out.astype(np.float32).reshape(-1)

def build_feature_from_mphand(landmarks) -> np.ndarray:
    row = []
    for lm in landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
    return np.asarray(row, dtype=np.float32)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loadedModel = torch.load(MODEL_PATH, map_location=device)

    input_dim   = int(loadedModel["input_dim"])
    num_classes = int(loadedModel["num_classes"])
    classes     = loadedModel.get("classes", [str(i) for i in range(num_classes)])
    add_vel     = bool(loadedModel.get("add_vel", False))

    net = DynamicGestureNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden=128,
        layers=2,
        bidir=True,
        dropout=0.0
    ).to(device)
    net.load_state_dict(loadedModel["model_state_dict"])
    net.eval()

    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
    mp_hands   = mp.solutions.hands

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    feat_window = deque(maxlen=WINDOW_SIZE)
    frame_count = 0
    labelString = None
    confidenceVal = 0.0

    prev_t = time.perf_counter()

    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Ignoring empty camera frame.")
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True

            #Draw landmarks
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if res.multi_hand_landmarks and DRAW_LANDMARKS:
                mp_drawing.draw_landmarks(
                    bgr,
                    res.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            #matches collected hand landmarks to GRU desired resolution
            if res.multi_hand_landmarks:
                scaledResult = build_feature_from_mphand(res.multi_hand_landmarks[0])
                scaledResult = _normalize_shape(scaledResult)
            else:
                #repeat last vector if available, otherwise zeros
                if len(feat_window) > 0:
                    scaledResult = feat_window[-1].copy()
                else:
                    scaledResult = np.zeros(63, dtype=np.float32)

            feat_window.append(scaledResult)

            #Send a sliding window of frames to the GRU
            frame_count += 1
            if len(feat_window) == WINDOW_SIZE and (frame_count % STRIDE == 0):
                seq = np.stack(feat_window, axis=0).astype(np.float32)
                if add_vel:
                    vel = np.diff(seq, axis=0, prepend=seq[:1]).astype(np.float32)
                    seq = np.concatenate([seq, vel], axis=1).astype(np.float32)

                x = torch.from_numpy(seq).unsqueeze(0).to(device)
                lengths = torch.tensor([seq.shape[0]], dtype=torch.long).to(device)

                with torch.no_grad():
                    logits = net(x, lengths)
                    probs = torch.softmax(logits, dim=1)
                    conf, idx = torch.max(probs, dim=1)
                    confidenceVal = float(conf.item())
                    pred_raw = classes[int(idx.item())]
                    labelString = decode_ipn(pred_raw) if confidenceVal >= MIN_CONF else None

            #Drawing hand landmarks
            disp = cv2.flip(bgr, 1)

            # Drawing FPS
            now = time.perf_counter()
            fps = 1.0 / (now - prev_t) if now > prev_t else 0.0
            prev_t = now
            cv2.putText(disp, f"{fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

            #Drawing the label
            if labelString:
                cv2.putText(disp, f"Dynamic: {labelString} ({confidenceVal:.2f})", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 3, cv2.LINE_AA)

            cv2.imshow("Dynamic Gesture Detection (GRU)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
