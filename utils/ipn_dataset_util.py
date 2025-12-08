import numpy as np

def augment_sequence(npArray: np.ndarray, p_rot=0.7, p_mask=0.4, p_warp=0.4, p_noise=0.8):
    T = npArray.shape[0]
    pts = npArray.reshape(T, 21, 3).astype(np.float32)

    #Rotation augmentation
    if np.random.rand() < p_rot:
        ang = np.deg2rad(np.random.uniform(-18.0, 18.0)).astype(np.float32)
        ca, sa = np.cos(ang), np.sin(ang)
        R = np.array([[ca, -sa, 0], [sa,  ca, 0], [0,   0, 1]], dtype=np.float32)
        pts = pts @ R.T

    if np.random.rand() < p_mask and T > 4:
        w = np.random.randint(2, min(8, T // 5) + 1)
        s = np.random.randint(1, T - w + 1)
        pts[s:s+w] = pts[s-1:s]

    #Slow or speed up movements
    if np.random.rand() < p_warp and T > 4:
        r = np.random.uniform(0.85, 1.15)
        t_orig = np.arange(T, dtype=np.float32)
        t_new = np.linspace(0, T-1, int(round(T * r)), dtype=np.float32)

        flat = pts.reshape(T, -1)
        warped = np.stack([np.interp(t_new, t_orig, flat[:, d])
                           for d in range(flat.shape[1])], axis=1)

        t_back = np.linspace(0, warped.shape[0] - 1, T, dtype=np.float32)
        pts = np.stack([np.interp(t_back,
                                  np.arange(warped.shape[0], dtype=np.float32),
                                  warped[:, d])
                        for d in range(warped.shape[1])], axis=1)
        pts = pts.reshape(T, 21, 3)

    #Introduce some randomization (noise)
    if np.random.rand() < p_noise:
        pts += np.random.normal(0.0, 0.015, pts.shape).astype(np.float32)

    return pts.reshape(T, 63).astype(np.float32)


class NPZSequenceDatasetCPU:
    #this data structure allows us to iterate through the IPN directory
    def __init__(self, files, labels_encoded, add_vel=False, train=False, augmenter=None):
        self.files = list(files)
        self.labels = np.asarray(labels_encoded, dtype=np.int64)
        self.add_vel = add_vel
        self.train = train
        self.augment = augmenter

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as npz:
            x = npz["x"].astype(np.float32)
            y = self.labels[idx]

        if self.train and self.augment is not None:
            x = self.augment(x)

        if self.add_vel:
            v = np.diff(x, axis=0, prepend=x[:1])
            x = np.concatenate([x, v], axis=1)

        return x, y, x.shape[0]
