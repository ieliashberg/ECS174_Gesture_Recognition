from pathlib import Path
import os
import csv
import cv2
import numpy as np
import mediapipe as mp
import time

# VID_PATH = Path("..\IPN\Videos")
# ANNS = Path("..\IPN\Annotations\Annot_List.txt")
# OUT_DIR = Path("datasets\IPN_dynamic")

# OUT_DIR.mkdir(parents = True, exists_ok=True)


VIDS = Path("..\IPN\Videos")
ANNS = Path("..\IPN\Annotations\Annot_List.txt")
OUT_DIR = Path("datasets\IPN_dynamic")

OUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_BASE = 1

EXTS = (".avi", ".AVI")

def resolve_video_path(token: str) -> Path | None:
    """
    Normalize the path from the CSV to an existing file.
    - If token has no extension, try adding .avi under VIDEO_ROOT.
    - If token is relative, anchor under VIDEO_ROOT.
    - Finally, search by stem anywhere under VIDEO_ROOT for a .avi.
    """
    token = str(token).strip().strip('"').strip("'")
    p = Path(token)

    # already absolute & exists?
    if p.is_absolute() and p.is_file():
        return p.resolve()

    # candidate under VIDEO_ROOT (as-given)
    cand = VIDS / p
    if cand.is_file():
        return cand.resolve()

    # if no extension, try .avi under VIDEO_ROOT
    if p.suffix == "":
        for ext in EXTS:
            hit = (VIDS / (p.name + ext))
            if hit.is_file():
                return hit.resolve()

    # as a last resort, search by stem for .avi
    stem = p.stem if p.suffix else p.name
    for ext in EXTS:
        for hit in VIDS.rglob(f"*{ext}"):
            if hit.stem == stem:
                return hit.resolve()

    return None

def _to_int(x, default=None):
    try:
        return int(str(x).strip())
    except Exception:
        return default

def parse_row(row):
    """
    Accept either:
      [video_path, label, start, end]  or
      [video_path, start, end, label]
    Return (video_token, start:int, end:int, label:str) or None.
    """
    row = [c.strip().strip('"').strip("'") for c in row]
    if len(row) < 4:
        return None
    path = row[0]; a, b, c = row[1], row[2], row[3]

    # case 1: label first, then numbers
    if not a.isdigit() and _to_int(b) is not None and _to_int(c) is not None:
        return path, _to_int(b), _to_int(c), a

    # case 2: numbers first, then label
    if _to_int(a) is not None and _to_int(b) is not None and not c.isdigit():
        return path, _to_int(a), _to_int(b), c

    # fallback: assume [path, start, end, label]
    s = _to_int(row[1]); e = _to_int(row[2]); lab = row[3]
    if s is None or e is None:
        return None
    return path, s, e, lab

def iter_annotations(annot_file: Path):
    """Yield (resolved_video_path, start0, end0, label) with indices normalized to 0-based."""
    with annot_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)  # comma-separated
        for row in reader:
            if not row:
                continue
            head = row[0].lstrip()
            if head.startswith("#") or head.lower().startswith(("video", "path")):
                continue
            parsed = parse_row(row)
            if not parsed:
                continue
            token, s, e, label = parsed
            # normalize to 0-based inclusive bounds
            s0 = max(0, (s or 0) - INDEX_BASE)
            e0 = (e or 0) - INDEX_BASE
            # resolve to an actual file (prefer .avi)
            vpath = resolve_video_path(token)
            if vpath is None:
                print("[WARN] could not resolve video:", token)
                continue
            yield vpath, s0, e0, label

def run_mediapipe_on_video(video_path: Path, segments):
    """
    segments: list of (s0, e0, label) with 0-based inclusive indices.
    Returns dict[(s0,e0,label)] -> np.ndarray (T,63).
    """
    results = {}
    if not segments:
        return results

    # Calc scan window and clamp to video length
    cap = cv2.VideoCapture(os.fspath(video_path))
    if not cap.isOpened():
        print("[WARN] cannot open:", video_path)
        return results
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    min_s = max(0, min(s for s, e, _ in segments))
    max_e = min(frame_count - 1, max(e for s, e, _ in segments))

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        # seek to first needed frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(min_s))
        frame_idx = min_s - 1
        buffers = {(s, e, lab): [] for (s, e, lab) in segments}

        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx > max_e:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out = hands.process(rgb)

            if out.multi_hand_landmarks:
                lm = out.multi_hand_landmarks[0].landmark
                vec = np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float32).reshape(-1)
            else:
                vec = np.full(63, np.nan, dtype=np.float32)

            for (s0, e0, lab) in segments:
                if s0 <= frame_idx <= e0:
                    buffers[(s0, e0, lab)].append(vec)

    cap.release()

    for key, seq in buffers.items():
        if seq:
            results[key] = np.stack(seq)  # (T,63)
    return results

def main():
    # group segments per video
    segs_by_video: dict[Path, list[tuple[int,int,str]]] = {}
    for vpath, s0, e0, label in iter_annotations(ANNS):
        if e0 < s0:
            continue
        segs_by_video.setdefault(vpath, []).append((s0, e0, label))

    # process each video once; save one .npz per segment
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for vpath, segs in segs_by_video.items():
        seqs = run_mediapipe_on_video(vpath, segs)
        for (s0, e0, label), arr in seqs.items():
            stem = vpath.stem.replace(".", "_")
            out = OUT_DIR / f"{stem}_{s0}_{e0}_{label}.npz"
            np.savez(out, x=arr, label=label)

if __name__ == "__main__":
    main()