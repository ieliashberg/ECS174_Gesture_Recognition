# ipn_dataset_util.py
from pathlib import Path
import numpy as np

def _rot_z(pts, max_deg=15):
    ang = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    ca, sa = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    R = np.array([[ca, -sa, 0.0], [sa,  ca, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (pts @ R.T).astype(np.float32)

def _scale_translate(pts, scale_range=(0.9, 1.1), trans_std=0.01):
    s = np.random.uniform(scale_range[0], scale_range[1])
    t = np.random.normal(0.0, trans_std, size=(1, 1, 3)).astype(np.float32)
    return (pts * np.float32(s) + t).astype(np.float32)

def _temporal_mask(pts, max_width=6):
    # mask a short time span by holding previous frame (SpecAugment-ish)
    T = pts.shape[0]
    if T <= 2: return pts
    w = np.random.randint(2, min(max_width, max(2, T // 5)) + 1)
    s = np.random.randint(1, T - w + 1)
    pts[s:s+w] = pts[s-1:s]
    return pts

def _time_warp_same_length(pts, low=0.9, high=1.1):
    # resample in time by factor r then back to original T via linear interp
    T = pts.shape[0]
    if T < 4: return pts
    r = np.random.uniform(low, high)
    t_orig = np.arange(T, dtype=np.float32)
    t_new  = np.linspace(0, T-1, num=int(round(T * r)), dtype=np.float32)
    # interp per feature
    D = pts.shape[1]*pts.shape[2]
    flat = pts.reshape(T, D)
    warped = np.empty((t_new.size, D), dtype=np.float32)
    for d in range(D):
        warped[:, d] = np.interp(t_new, t_orig, flat[:, d]).astype(np.float32)
    # back to length T
    t_back = np.linspace(0, warped.shape[0]-1, num=T, dtype=np.float32)
    back = np.empty((T, D), dtype=np.float32)
    for d in range(D):
        back[:, d] = np.interp(t_back, np.arange(warped.shape[0], dtype=np.float32), warped[:, d]).astype(np.float32)
    return back.reshape(T, pts.shape[1], pts.shape[2])

def _gaussian_noise(pts, std=0.01):
    return (pts + np.random.normal(0.0, std, size=pts.shape).astype(np.float32)).astype(np.float32)

def augment_sequence(x63: np.ndarray,p_rot=0.9, p_st=0.9, p_mask=0.5, p_warp=0.3, p_noise=0.7, enable_mirror=False):
    x = x63.astype(np.float32, copy=False)
    T = x.shape[0]
    pts = x.reshape(T, 21, 3).astype(np.float32)

    if np.random.rand() < p_rot:
        pts = _rot_z(pts, max_deg=15)
    if np.random.rand() < p_st:
        pts = _scale_translate(pts, scale_range=(0.9, 1.1), trans_std=0.01)
    if np.random.rand() < p_mask:
        pts = _temporal_mask(pts, max_width=6)
    if np.random.rand() < p_warp:
        pts = _time_warp_same_length(pts, low=0.9, high=1.1)
    if np.random.rand() < p_noise:
        pts = _gaussian_noise(pts, std=0.01)

    if enable_mirror and np.random.rand() < 0.5:
        pts[..., 0] = -pts[..., 0]

    return pts.reshape(T, 63).astype(np.float32)

class NPZSequenceDatasetCPU:
    def __init__(self, files, labels_encoded, add_vel=False, normalizer=None, train = False, augmenter=None):
        self.files = list(files)
        self.y     = np.asarray(labels_encoded, dtype=np.int64)
        self.add_vel = add_vel
        self.normalizer = normalizer
        self.train = train
        self.augmenter = augmenter

    def __len__(self):
        return len(self.files)

    def _forward_fill_nans(self, x):
        mask = np.isnan(x)
        if not np.any(mask):
            return x
        idx = np.where(~mask, np.arange(x.shape[0])[:, None], 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        x = x[idx, np.arange(x.shape[1])]
        x[mask] = 0.0
        return x

    def __getitem__(self, i):
        with np.load(self.files[i]) as npz:
            x = npz["x"].astype(np.float32)
            y = self.y[i]

        x = self._forward_fill_nans(x)
        if self.normalizer is not None:
            x = self.normalizer(x)

        if self.train and self.augmenter is not None:
            x = self.augmenter(x).astype(np.float32, copy=False)

        if self.add_vel:
            v = np.diff(x, axis=0, prepend=x[:1])
            x = np.concatenate([x, v], axis=1)

        return x, y, x.shape[0]

def collate_np(batch):
    xs, ys, lens = zip(*batch)
    B = len(xs)
    lens = np.asarray(lens, dtype=np.int64)
    D = xs[0].shape[1]
    T_max = int(max(lens))
    out = np.zeros((B, T_max, D), dtype=np.float32)
    for i, x in enumerate(xs):
        t = x.shape[0]
        out[i, :t, :] = x
    ys = np.asarray(ys, dtype=np.int64)
    return out, lens, ys
