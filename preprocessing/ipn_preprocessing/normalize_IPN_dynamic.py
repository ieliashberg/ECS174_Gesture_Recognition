"""
Normalize IPN landmark .npz files using existing normalization logic.

Pipeline:
1) Load dynamic gesture file (T,64) where last dim is presence flag
2) Drop the presence flag (T,63)
3) Remove NaN-only boundary frames
4) Detect and drop sequences with internal NaN gaps > MAX_GAP
5) Forward-fill small NaN gaps
6) Normalize using your existing normalize() pipeline
7) Save to IPN_dynamic_npz_normalized
"""

from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

#Make project root importable so we can use utils.normalize
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.utils import normalize


DATASETS = Path("../../datasets/IPN")
SRC = DATASETS / "IPN_dynamic_npz"
DST = DATASETS / "IPN_dynamic_npz_normalized"
MIN_FRAMES = 10
MAX_GAP = 3

#Some helper functions
def extract_label(filename: str) -> str:
    return filename.split("_")[-3]


def remove_nan_borders(x: np.ndarray) -> np.ndarray | None:
    """
    Remove NaN-only frames at the start/end; keep interior NaNs for later fix.
    Returns a possibly-shorter array, or None if there are no valid frames.
    """
    mask = ~np.isnan(x).any(axis=1)
    if not np.any(mask):
        return None
    first = np.argmax(mask)
    last = len(mask) - np.argmax(mask[::-1])
    return x[first:last]


def find_long_gaps(mask: np.ndarray) -> bool:
    """
    Given a boolean mask of valid frames (True = valid),
    return True if there is a gap of invalid frames > MAX_GAP.
    """
    idx = np.where(mask)[0]
    if len(idx) <= 1:
        return True
    return np.diff(idx).max() > MAX_GAP


def forward_fill_nans(x: np.ndarray) -> np.ndarray:
    """Replace missing frames with previous frame (forward fill)."""
    mask = np.isnan(x)
    if not np.any(mask):
        return x
    idx = np.where(~mask, np.arange(x.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    x = x[idx, np.arange(x.shape[1])]
    x[mask] = 0.0
    return x

#Main loop
def normalize_all():
    DST.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.npz"))

    kept = dropped_short = dropped_gap = 0

    used_remove_nan_borders = 0
    used_find_long_gaps = 0
    used_forward_fill_nans = 0

    #how many kept files actually had their data changed by remove_nan_borders or forward_fill_nans
    changed_files = 0

    print(f"Normalizing {len(files)} gesture clips...\n")

    for f in tqdm(files, ncols=90):
        with np.load(f) as npz:
            data = npz["data"].astype(np.float32)

        x = data[:, :63]
        label = extract_label(f.stem)

        #Keep a copy to detect changes
        x_orig = x.copy()

        used_remove_nan_borders += 1
        x = remove_nan_borders(x)
        if x is None or x.shape[0] < MIN_FRAMES:
            dropped_short += 1
            continue

        if x.shape != x_orig.shape or not np.array_equal(x, x_orig):
            trimmed_changed = True
        else:
            trimmed_changed = False

        mask = ~np.isnan(x).any(axis=1)
        used_find_long_gaps += 1
        if find_long_gaps(mask):
            dropped_gap += 1
            continue

        x_before_fill = x.copy()
        used_forward_fill_nans += 1
        x = forward_fill_nans(x)

        if x.shape != x_before_fill.shape or not np.array_equal(x, x_before_fill):
            fill_changed = True
        else:
            fill_changed = False

        #this file's data changed if either trimming or fill had an effect
        this_file_changed = trimmed_changed or fill_changed

        x = normalize(x)

        np.savez(DST / f.name, x=x.astype(np.float32), label=label)
        kept += 1

        if this_file_changed:
            changed_files += 1

    print("\nDONE.")
    print(f"Source files                        : {len(files)}")
    print(f"Kept (written out)                  : {kept}")
    print(f"Dropped (short)                     : {dropped_short}")
    print(f"Dropped (gaps)                      : {dropped_gap}")
    print("-------------------------------------------------")
    print(f"remove_nan_borders called           : {used_remove_nan_borders}")
    print(f"find_long_gaps called               : {used_find_long_gaps}")
    print(f"forward_fill_nans called            : {used_forward_fill_nans}")
    print("-------------------------------------------------")
    print(f"Total .npz files changed by cleaning: {changed_files}")


if __name__ == "__main__":
    normalize_all()
