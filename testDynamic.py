# dynamic_main.py
import cv2
import time
import numpy as np
import torch
from collections import deque
import mediapipe as mp

from dynamic_gesture_net import DynamicGestureNet
from utils import normalize

# ------------- Settings -------------
MODEL_PATH = "trained_models\dynamic_gesture_net_v1.pt"  # your GRU checkpoint path
CAM_INDEX = 0
WINDOW_SIZE = 30
STRIDE = 15
MIN_CONF = 0.85
DRAW_LANDMARKS = True
# Optional: map IPN codes => friendly labels
IPN_DECODE = {
    "G01": "Swipe Left",
    "G02": "Swipe Right",
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
# ------------------------------------

def decode_ipn(lbl: str) -> str:
    return IPN_DECODE.get(lbl, lbl)

def _normalize_63(vec63: np.ndarray) -> np.ndarray:
    v = np.asarray(vec63, dtype=np.float32).reshape(1, -1)
    out = normalize(v)
    return out.astype(np.float32).reshape(-1)

def build_feature_from_mphand(landmarks) -> np.ndarray:
    """Flatten MediaPipe 21 landmarks -> (63,) [x0,y0,z0,...,x20,y20,z20]."""
    row = []
    for lm in landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
    return np.asarray(row, dtype=np.float32)

def main():
    # ---------- Load GRU checkpoint ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)

    input_dim   = int(ckpt["input_dim"])
    num_classes = int(ckpt["num_classes"])
    classes     = ckpt.get("classes", [str(i) for i in range(num_classes)])
    add_vel     = bool(ckpt.get("add_vel", False))

    net = DynamicGestureNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden=128,
        layers=2,
        bidir=True,
        dropout=0.0
    ).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    # ---------- MediaPipe Hands ----------
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
    mp_hands   = mp.solutions.hands

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    feat_window = deque(maxlen=WINDOW_SIZE)
    frame_count = 0
    dyn_label_str = None
    dyn_conf = 0.0

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
                print("Ignoring empty camera frame.")
                continue

            # BGR -> RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True

            # Draw landmarks (optional)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if res.multi_hand_landmarks and DRAW_LANDMARKS:
                mp_drawing.draw_landmarks(
                    bgr,
                    res.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            # ---- Per-frame 63-D features ----
            if res.multi_hand_landmarks:
                vec63 = build_feature_from_mphand(res.multi_hand_landmarks[0])
                vec63 = _normalize_63(vec63)
            else:
                # If no detection: repeat last vector if available, otherwise zeros
                if len(feat_window) > 0:
                    vec63 = feat_window[-1].copy()
                else:
                    vec63 = np.zeros(63, dtype=np.float32)

            feat_window.append(vec63)

            # ---- Sliding-window GRU inference ----
            frame_count += 1
            if len(feat_window) == WINDOW_SIZE and (frame_count % STRIDE == 0):
                seq = np.stack(feat_window, axis=0).astype(np.float32)  # (T,63)
                if add_vel:
                    vel = np.diff(seq, axis=0, prepend=seq[:1]).astype(np.float32)
                    seq = np.concatenate([seq, vel], axis=1).astype(np.float32)  # (T,126)

                x = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1,T,D)
                lengths = torch.tensor([seq.shape[0]], dtype=torch.long).to(device)

                with torch.no_grad():
                    logits = net(x, lengths)
                    probs = torch.softmax(logits, dim=1)
                    conf, idx = torch.max(probs, dim=1)
                    dyn_conf = float(conf.item())
                    pred_raw = classes[int(idx.item())]
                    dyn_label_str = decode_ipn(pred_raw) if dyn_conf >= MIN_CONF else None

            # ---- UI overlays ----
            disp = cv2.flip(bgr, 1)  # mirror for user

            # FPS
            now = time.perf_counter()
            fps = 1.0 / (now - prev_t) if now > prev_t else 0.0
            prev_t = now
            cv2.putText(disp, f"{fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

            # Dynamic label
            if dyn_label_str:
                cv2.putText(disp, f"Dynamic: {dyn_label_str} ({dyn_conf:.2f})", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 3, cv2.LINE_AA)

            cv2.imshow("Dynamic Gesture Detection (GRU)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
