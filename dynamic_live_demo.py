# dynamic_main.py
import cv2
import time
import numpy as np
import torch
from collections import deque
import mediapipe as mp

from model_definitions.dynamic_gesture_netv2 import DynamicGestureNet
from utils.utils import normalize

MODEL_PATH = "trained_models/dynamic_gesture_netv2_best.pt"
CAM_INDEX = 0

# ===== REAL-TIME CONTROL PARAMETERS =====
WINDOW_SIZE      = 45    # reduced from 65 → MUCH lower latency
STRIDE           = 3     # process every 3rd frame
MIN_CONF         = 0.70  # require high confidence
MOTION_THRESHOLD = 0.025 # minimal hand motion to allow prediction
COOLDOWN         = 3    # frames to silence after gesture detection
RECENT_MOTION_T  = 3     # how many latest frames to compute motion over
DRAW_LANDMARKS   = True

IPN_DECODE = {
    "G01": "Click with one finger", "G02": "Click with two fingers",
    "D0X": "no_gesture", "B0A": "Pointing with one finger",
    "B0B": "Pointing with two fingers", "G03": "Throw up",
    "G04": "Throw down", "G05": "Throw left", "G06": "Throw right",
    "G07": "Open twice", "G08": "Double click with one finger",
    "G09": "Double click with two fingers", "G10": "Zoom in",
    "G11": "Zoom out",
}

def decode(label): return IPN_DECODE.get(label, label)

def normalize_feat(vec):
    """Apply SAME normalization as training."""
    v = np.asarray(vec, np.float32).reshape(1, -1)
    return normalize(v).reshape(-1).astype(np.float32)

def build_feature(hand_lms):
    """Convert mediapipe hand→63-dim vector"""
    out = []
    for lm in hand_lms.landmark:
        out.extend([lm.x, lm.y, lm.z])
    return np.asarray(out, np.float32)

# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)

    input_dim   = ckpt["input_dim"]
    num_classes = ckpt["num_classes"]
    classes     = ckpt["classes"]
    add_vel     = ckpt["add_vel"]

    net = DynamicGestureNet(input_dim, num_classes, hidden=128, layers=2, bidir=True, dropout=0.0).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Camera not available."); return

    feat_window = deque(maxlen=WINDOW_SIZE)
    frame_idx   = 0
    cooldown    = 0
    last_pred   = None
    prev_t      = time.perf_counter()

    with mp_hands.Hands(model_complexity=1, max_num_hands=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

        while True:
            ok, frame = cap.read()
            if not ok: continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # draw landmarks
            if res.multi_hand_landmarks and DRAW_LANDMARKS:
                mp_draw.draw_landmarks(bgr, res.multi_hand_landmarks[0],
                                       mp_hands.HAND_CONNECTIONS,
                                       mp_style.get_default_hand_landmarks_style(),
                                       mp_style.get_default_hand_connections_style())

            # build normalized feature
            if res.multi_hand_landmarks:
                feat = normalize_feat(build_feature(res.multi_hand_landmarks[0]))
            else:
                feat = feat_window[-1].copy() if len(feat_window) else np.zeros(63, np.float32)

            feat_window.append(feat)
            frame_idx += 1

            # ---------- MOTION GATING ----------
            if len(feat_window) > 1:
                seq_arr = np.stack(feat_window, axis=0)
                vel = np.diff(seq_arr[-RECENT_MOTION_T:], axis=0) if len(seq_arr) > RECENT_MOTION_T else np.diff(seq_arr, axis=0)
                motion = float(np.mean(np.linalg.norm(vel, axis=1)))
            else:
                motion = 0.0

            # ---------- PREDICT ----------
            if cooldown == 0 and len(feat_window) == WINDOW_SIZE and frame_idx % STRIDE == 0:
                seq = np.stack(feat_window, 0).astype(np.float32)

                if add_vel:
                    v = np.diff(seq, axis=0, prepend=seq[:1])
                    seq = np.concatenate([seq, v], axis=1)

                x = torch.from_numpy(seq).unsqueeze(0).to(device)
                lengths = torch.tensor([seq.shape[0]], dtype=torch.long, device=device)

                with torch.no_grad():
                    probs = torch.softmax(net(x, lengths), dim=1)
                    conf, idx = torch.max(probs, 1)
                    conf = float(conf)
                    pred = classes[int(idx)]

                if conf >= MIN_CONF and motion >= MOTION_THRESHOLD:
                    last_pred = decode(pred)
                    cooldown = COOLDOWN               # freeze output
                    feat_window.clear()               # 💥 flush gesture history
                else:
                    last_pred = None

            else:
                if cooldown > 0:
                    cooldown -= 1
                    last_pred = None  # silence during cooldown

            # ---------- DRAW UI ----------
            disp = cv2.flip(bgr, 1)
            now = time.perf_counter()
            fps = 1.0 / (now - prev_t)
            prev_t = now
            cv2.putText(disp, f"{fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

            if last_pred:
                cv2.putText(disp, f"{last_pred}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 3)

            cv2.imshow("Dynamic Gesture Detection", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
