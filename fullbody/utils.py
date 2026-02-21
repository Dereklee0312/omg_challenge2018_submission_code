from pathlib import Path
import re

import numpy as np
import tensorflow.keras.backend as K
import keras.callbacks as cb

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize

from scipy.stats import pearsonr


import time

day_time = time.strftime("%Y-%m-%d_%H_%M_%S")


def ccc(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)

    rho, _ = pearsonr(y_pred, y_true)
    std_predictions = np.std(y_pred)
    std_gt = np.std(y_true)

    ccc = (
        2
        * rho
        * std_gt
        * std_predictions
        / (std_predictions**2 + std_gt**2 + (pred_mean - true_mean) ** 2 + 1e-8)
    )

    return ccc, rho


def ccc_error(y_true, y_pred):
    true_mean = K.mean(y_true)
    pred_mean = K.mean(y_pred)

    x = y_true - true_mean
    y = y_pred - pred_mean
    rho = K.sum(x * y) / (K.sqrt(K.sum(x**2) * K.sum(y**2)) + K.epsilon())

    std_predictions = K.std(y_pred)
    std_gt = K.std(y_true)

    ccc = (
        2
        * rho
        * std_gt
        * std_predictions
        / (std_predictions**2 + std_gt**2 + (pred_mean - true_mean) ** 2 + K.epsilon())
    )
    return 1 - ccc


class Metrics(cb.Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(self.model.predict(X_val, verbose=0))

        ccc_result, rho_result = ccc(y_val, y_predict)

        self._data.append({"ccc": ccc_result, "rho": rho_result})
        print("ccc = %f,  pearson=%f" % (ccc_result, rho_result))
        return

    def get_data(self):
        return self._data


def moving_avg(x, win=300):
    x_av = np.zeros(len(x))

    for t in range(len(x)):
        x_av[t] = np.mean(x[t : t + win])

    return x_av


def moving_avg_ctr(x, win=300):
    x_av = np.zeros(len(x))

    for t in range(int(win / 2), int(len(x) - win / 2)):
        x_av[t] = np.mean(x[t - int(win / 2) : t + int(win / 2)])

    return x_av


def norm_pred(lbl, pred):
    s0 = np.std(lbl.flatten())
    V = pred.flatten()
    m1 = np.mean(pred.flatten())
    s1 = np.std(pred.flatten())
    m0 = np.mean(lbl.flatten())

    norm_pred = s0 * (V - m1) / s1 + m0

    return norm_pred


def create_img_vec(img_path, sbj_n, str_n, down_sampling, img_x, img_y):
    path = Path(img_path.format(sbj_n, str_n, subject=sbj_n, story=str_n))
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    frames_n = [p.name for p in path.iterdir() if p.is_file()]
    indexed = []
    for name in frames_n:
        match = re.search(r"(\d+)", name)
        if match:
            indexed.append((int(match.group(1)), name))
    indexed.sort(key=lambda item: item[0])
    sorted_frames_n = [name for _, name in indexed[::down_sampling]]
    if not sorted_frames_n:
        raise ValueError(f"No usable frames found in {path}")

    img_s = []

    for f_n in sorted_frames_n:
        iii = io.imread(path / f_n)
        if iii.ndim == 3 and iii.shape[-1] != 1:
            iii = rgb2gray(iii)
        elif iii.ndim == 2:
            pass
        else:
            # fallback: squeeze any extra channels
            iii = np.squeeze(iii)
            if iii.ndim == 3:
                iii = rgb2gray(iii)
        if iii.shape != (img_x, img_y):
            iii = resize(iii, (img_x, img_y), mode="reflect", anti_aliasing=True)
        iii = iii.astype(np.float32)
        mean = np.mean(iii)
        std = np.std(iii)
        iii = (iii - mean) if std == 0 else (iii - mean) / std
        img_s.append(iii.reshape(img_x, img_y, 1))

    return np.array(img_s)


import numpy as np
from scipy.signal import butter, lfilter


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def padder(fff, pad_len, pad_side="f"):
    if pad_side == "f":
        pad = np.vstack([fff[0, :] for i in range(pad_len)])
        pad_fff = np.vstack([pad, fff])

    elif pad_side == "b":
        pad = np.vstack([fff[-1, :] for i in range(pad_len)])
        pad_fff = np.vstack([fff, pad])

    return pad_fff


def expand_pred(x, exp_rate=5):
    y = np.zeros([x.shape[0] * exp_rate, 1])
    ii = 0

    for p in x:
        y[ii : ii + exp_rate] = np.ones([exp_rate, 1]) * p
        ii += exp_rate

    return y


def sequence_reshape(img, lbl, seq_len):
    img_resh = []
    lbl_resh = []

    for i in range(img.shape[0] - seq_len):
        img_resh.append(img[i : i + seq_len, :, :, :])
        lbl_resh.append(lbl[i + seq_len, :])

    lbl_resh = np.array(lbl_resh)
    img_resh = np.array(img_resh)

    print("image shape: ", img_resh.shape)
    print("label shape: ", lbl_resh.shape)

    return img_resh, lbl_resh


class light_generator:
    def __init__(self, x, y, seq_len, batch_size):
        self.x = x
        self.y = y

        self.seq_len = seq_len
        self.sample_size = self.x.shape[0]

        self.h = self.x.shape[1]
        self.w = self.x.shape[2]
        self.c = self.x.shape[3]

        self.idx_s = np.arange(self.sample_size - self.seq_len)
        self.batch_size = batch_size
        self.stp_per_epoch = max(
            1, int(np.ceil(len(self.idx_s) / float(self.batch_size)))
        )

    def generate(self):
        while True:
            for b in range(self.stp_per_epoch):
                np.random.shuffle(self.idx_s)
                rnd_idx = self.idx_s[: self.batch_size]
                current_bs = len(rnd_idx)

                xb = np.empty(
                    [current_bs, self.seq_len, self.h, self.w, self.c],
                    dtype=self.x.dtype,
                )
                yb = np.empty([current_bs, 1], dtype=self.y.dtype)

                for i, ri in enumerate(rnd_idx):
                    xb[i, ...] = self.x[ri : ri + self.seq_len, :, :, :]
                    yb[i, :] = self.y[ri + self.seq_len, :]

                yield xb, yb


def create_img_dataset(
    img_path, img_x, img_y, ch_n, str_n_s, sbj_n_s, down_sampling=5
):
    img_slices = []
    for str_n in str_n_s:
        for sbj_n in sbj_n_s:
            img_vec = create_img_vec(img_path, sbj_n, str_n, down_sampling, img_x, img_y)
            img_slices.append(img_vec)
            print(f"loaded images for story {str_n}, subject {sbj_n}: {img_vec.shape}")

    if not img_slices:
        raise ValueError("No images loaded; check image paths and templates.")

    return np.concatenate(img_slices, axis=0)
