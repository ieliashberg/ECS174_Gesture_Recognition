# ipn_dataset_util.py
from pathlib import Path
import numpy as np

class NPZSequenceDatasetCPU:
    def __init__(self, files, labels_encoded, add_vel=False, normalizer=None):
        self.files = list(files)
        self.y     = np.asarray(labels_encoded, dtype=np.int64)
        self.add_vel = add_vel
        self.normalizer = normalizer  # pass a callable that accepts (T,D) np and returns (T,D) np

    def __len__(self):
        return len(self.files)

    def _forward_fill_nans(self, x):
        # x: (T, D) float32 with possible NaNs
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
            x = npz["x"].astype(np.float32)   # (T,63)
            y = self.y[i]

        x = self._forward_fill_nans(x)
        if self.normalizer is not None:
            x = self.normalizer(x)            # (T,63), pure NumPy

        if self.add_vel:
            v = np.diff(x, axis=0, prepend=x[:1])
            x = np.concatenate([x, v], axis=1)  # (T,126)

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
