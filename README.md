## ECS 174 – Hand Gesture Detection

Minimal PyTorch implementation of static and dynamic hand gesture recognition using MediaPipe landmarks.

---

## Overview

- **Static model**: `StaticGestureNet` (MLP) trained on single-frame landmarks from `datasets/static_dataset.csv`.
- **Dynamic model**: `DynamicGestureNet` (GRU-based, with and without attention) trained on IPN-style sequences in `datasets/IPN/IPN_dynamic_npz_normalized/`.
- **Demos**:
  - `dynamic_live_demo.py`: real-time dynamic gesture classification from webcam.
  - `Prototype_BrowserControl.py`: static gesture control of browser via `pyautogui`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pyautogui            # for Prototype_BrowserControl.py
```

Datasets are not included. Place:
- Static CSV at `datasets/static_dataset.csv` (63 landmark values + label).
- Dynamic `.npz` clips under `datasets/IPN/IPN_dynamic_npz_normalized/` as expected by `utils/ipn_dataset_util.py`.

---

## Training

- **Static**:

  ```bash
  python training_scripts/train_static_model.py
  ```

  Uses `utils.normalize_data_file`, splits into train/val/test, and writes `static_gesture_net_v1.pt`.

- **Dynamic**:

  ```bash
  python training_scripts/train_dynamic_model.py
  ```

  Uses `NPZSequenceDatasetCPU`, attention-based `DynamicGestureNet`, and writes `dynamic_gesture_net_best.pt` and `dynamic_gesture_net_v11.pt`.

---

## Demos

- **Dynamic webcam demo**:

  ```bash
  python dynamic_live_demo.py
  ```

  Uses `trained_models/dynamic_gesture_net_v8.pt` by default. Adjust `MODEL_PATH` and `CAM_INDEX` as needed.

- **Static browser control**:

  ```bash
  python Prototype_BrowserControl.py
  ```

  Uses `trained_models/static_gesture_net_v3.pt` by default. Edit `KEYBINDINGS`, `ENABLE_LABEL`, and `DISABLE_LABEL` inside the script.



