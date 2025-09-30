#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import json
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.stats import pearsonr
from scipy.signal import *
import pandas as pd
import pickle
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.layers import Embedding, Dropout, LSTM, Bidirectional, concatenate, Reshape, Flatten
from keras import optimizers
import tensorflow.keras.backend as K
import time
from models.attlayer import AttentionWeightedAverage

# Parameters
subjects = [1]
stories_train = [1]
stories_val = [1]
normalize_labels = True
smooth = 0
modalities = ["text"]
base_path = "./data/"
labels_path = "./data/original_dataset/annotations/"
checkpoint_filename = "tmp_weights.h5"
batch_size = 500
epochs = 1000
patience = 5
lr = 0.0001
embedding_size = 11
window_size = 100
stride = 50
initial_dropout = 0.2
subject_vector_early = False
subject_vector_late = True
subject_vector_size = 2
lstm_stateful = False
lstm_units = 64
lstm_dropout = 0.2
activation = "relu"
lstm_output_dim = 64
lstm_attention = True
final_dropout = 0.2
lstm_attention_type = "softmax"
second_last_dim = 32
day_time = time.strftime("%Y-%m-%d_%H_%M_%S")

# Data
def get_X(story, subject, modality):
    file_name = "/Subject_" + str(subject) + "_Story_" + str(story) + "_aligned.npy"
    base_path = "./vectors/val2/"
    latent_vecs_path = base_path + "text_aligned" + file_name
    try:
        X = np.load(latent_vecs_path)
        return X
    except FileNotFoundError:
        print(f"Missing file: {latent_vecs_path}")
        return np.zeros((window_size, embedding_size))

def get_Y(story, subject, smooth=0):
    file_name = "/Subject_" + str(subject) + "_Story_" + str(story) + ".csv"
    labels_path_full = labels_path + file_name
    try:
        Y = open(labels_path_full).read().split("\n")[1:]
        Y = [float(x) for x in Y]
        return Y
    except FileNotFoundError:
        print(f"Missing file: {labels_path_full}")
        return np.zeros(100)

# utilities
def ccc(y_true, y_pred):
    true_mean = np.mean(y_true)
    true_variance = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_variance = np.var(y_pred)
    rho, _ = pearsonr(y_pred, y_true)
    std_predictions = np.std(y_pred)
    std_gt = np.std(y_true)
    ccc = 2 * rho * std_gt * std_predictions / (
       std_predictions ** 2 + std_gt ** 2 +
       (pred_mean - true_mean) ** 2)
    return ccc, rho

# loss (ccc implemented with tensors)
def ccc_error(y_true, y_pred):
    true_mean = K.mean(y_true)
    true_variance = K.var(y_true)
    pred_mean = K.mean(y_pred)
    pred_variance = K.var(y_pred)
    x = y_true - true_mean
    y = y_pred - pred_mean
    rho = K.sum(x * y) / K.sqrt(K.sum(x**2) * K.sum(y**2))
   
    std_predictions = K.std(y_pred)
    std_gt = K.std(y_true)
    ccc = 2 * rho * std_gt * std_predictions / (
       std_predictions ** 2 + std_gt ** 2 +
       (pred_mean - true_mean) ** 2)
    return 1 - ccc

# model
def build_model():
    input_late_subject = Input(shape=(1,), dtype='int32')
    input_lstm = Input(shape=(window_size, embedding_size))
    seq_input_drop = Dropout(initial_dropout)(input_lstm)
    if lstm_attention:
        lstm_output = LSTM(lstm_output_dim, return_sequences=True)(seq_input_drop)
        lstm_output, _ = AttentionWeightedAverage(name='attlayer')(lstm_output)
    else:
        lstm_output = LSTM(lstm_output_dim, return_sequences=False)(seq_input_drop)
   
    subject_embedding = Embedding(11, subject_vector_size)(input_late_subject)
    subject_embedding = Flatten()(subject_embedding)
    subject_concat = concatenate([subject_embedding, lstm_output])
    second_last = Dense(second_last_dim, name="second_last", activation=activation)(subject_concat)
    second_last = Dropout(final_dropout)(second_last)
    outputs = Dense(1)(second_last)
   
    return Model(inputs=[input_late_subject, input_lstm], outputs=outputs)

# measure ccc when epoch ends
class Metrics(keras.callbacks.Callback):
    def __init__(self, x_val_late_subject, x_val, y_val):
        super(Metrics, self).__init__()
        self.x_val_late_subject = x_val_late_subject
        self.x_val = x_val
        self.y_val = y_val
        self._data = []

    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        y_predict = np.asarray(self.model.predict([self.x_val_late_subject, self.x_val])).flatten()
        ccc_result, rho_result = ccc(self.y_val, y_predict)
       
        self._data.append({
            'ccc': ccc_result,  # CHANGED: Store scalar values
            'rho': rho_result
        })
        print("ccc = %f, pearson=%f" % (ccc_result, rho_result))  # CHANGED: Remove [0] indexing
        return

    def get_data(self):
        return self._data

# post processing functions
def f_trick(Y_train, preds):
    Y_train_flat = Y_train.flatten()
    preds_flat = preds.flatten()
    s0 = np.std(Y_train_flat)
    V = preds_flat
    m1 = np.mean(preds_flat)
    s1 = np.std(preds_flat)
    m0 = np.mean(Y_train_flat)
    norm_preds = s0*(V-m1)/s1+m0
    return norm_preds

def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter_bidirectional(data, cutoff=0.1, fs=25, order=1):
    y_first_pass = butter_lowpass_filter(data[::-1].flatten(), cutoff, fs, order)
    y_second_pass = butter_lowpass_filter(y_first_pass[::-1].flatten(), cutoff, fs, order)
    return y_second_pass

# Training data loading
print("-- Loading training data --")

# Collect all frame-level Y for global smoothing/normalization
Y_frame_list_train = []
frame_lengths_train = []
for subject in subjects:
    for story in stories_train:
        Y = get_Y(story, subject)
        Y_frame_list_train.append(Y)
        frame_lengths_train.append(len(Y))

Y_concat_train = np.concatenate(Y_frame_list_train)
if smooth > 0:
    Y_concat_train = butter_lowpass_filter_bidirectional(Y_concat_train, cutoff=smooth, fs=25, order=1)
if normalize_labels:
    min_y = np.min(Y_concat_train)
    max_y = np.max(Y_concat_train)
    Y_concat_train = (Y_concat_train - min_y) / (max_y - min_y + 1e-10)

# Split normalized/smoothed Y back to per story
start = 0
Y_parts_train = []
for l in frame_lengths_train:
    Y_parts_train.append(Y_concat_train[start:start + l])
    start += l

# Now load X and window both X and Y per story
X_windowed_train = []
Y_windowed_train = []
subjects_windowed_train = []
story_idx = 0
for sub_idx, subject in enumerate(subjects):
    for story in stories_train:
        X = get_X(story, subject, "text")
        Y_part = Y_parts_train[story_idx]
        num_frames = len(Y_part)
        if num_frames < window_size:
            story_idx += 1
            continue
        num_windows = ((num_frames - window_size) // stride) + 1
        for i in range(num_windows):
            start_i = i * stride
            end_i = start_i + window_size
            X_win = X[start_i:end_i]
            Y_win = np.mean(Y_part[start_i:end_i])
            X_windowed_train.append(X_win)
            Y_windowed_train.append(Y_win)
            subjects_windowed_train.append(sub_idx)
        story_idx += 1

X_train = np.array(X_windowed_train)
Y_train = np.array(Y_windowed_train)
X_train_late_subject = np.array(subjects_windowed_train)

# Validation data loading
print("-- Loading validation data --")

# Collect all frame-level Y for global smoothing/normalization
Y_frame_list_val = []
frame_lengths_val = []
for subject in subjects:
    for story in stories_val:
        Y = get_Y(story, subject)
        Y_frame_list_val.append(Y)
        frame_lengths_val.append(len(Y))

Y_concat_val = np.concatenate(Y_frame_list_val)
if smooth > 0:
    Y_concat_val = butter_lowpass_filter_bidirectional(Y_concat_val, cutoff=smooth, fs=25, order=1)
if normalize_labels:
    min_y = np.min(Y_concat_val)
    max_y = np.max(Y_concat_val)
    Y_concat_val = (Y_concat_val - min_y) / (max_y - min_y + 1e-10)

# Split normalized/smoothed Y back to per story
start = 0
Y_parts_val = []
for l in frame_lengths_val:
    Y_parts_val.append(Y_concat_val[start:start + l])
    start += l

# Now load X and window both X and Y per story
X_windowed_val = []
Y_windowed_val = []
subjects_windowed_val = []
story_idx = 0
for sub_idx, subject in enumerate(subjects):
    for story in stories_val:
        X = get_X(story, subject, "text")
        Y_part = Y_parts_val[story_idx]
        num_frames = len(Y_part)
        if num_frames < window_size:
            story_idx += 1
            continue
        num_windows = ((num_frames - window_size) // stride) + 1
        for i in range(num_windows):
            start_i = i * stride
            end_i = start_i + window_size
            X_win = X[start_i:end_i]
            Y_win = np.mean(Y_part[start_i:end_i])
            X_windowed_val.append(X_win)
            Y_windowed_val.append(Y_win)
            subjects_windowed_val.append(sub_idx)
        story_idx += 1

X_val = np.array(X_windowed_val)
Y_val = np.array(Y_windowed_val)
X_val_late_subject = np.array(subjects_windowed_val)

# Sanity checks
print("train lexicon vectors:", X_train.shape)
print("train subject late:", len(X_train_late_subject))
print("train labels:", len(Y_train))
print("val lexicon vectors:", X_val.shape)
print("val subject late:", len(X_val_late_subject))
print("val labels:", len(Y_val))

# Training
opt = optimizers.Adam(learning_rate=lr)
model = build_model()
model.compile(loss=ccc_error,
              optimizer=opt)
metrics = Metrics(X_val_late_subject, X_val, Y_val)
callbacks_list = [
    metrics,
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),
    keras.callbacks.ModelCheckpoint(filepath=checkpoint_filename, monitor='val_loss', save_best_only=True),
    keras.callbacks.TensorBoard(log_dir="../logs/lexicons_" + day_time)
]

history = model.fit([X_train_late_subject, X_train], Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=([X_val_late_subject, X_val], Y_val),
                    callbacks=callbacks_list)

# Plot predictions
model.load_weights(checkpoint_filename)
preds = model.predict([X_val_late_subject, X_val])
preds_tricks = f_trick(Y_train, preds)
ccc_result = ccc(Y_val, preds.flatten())
ccc_result_tricks = ccc(Y_val, preds_tricks.flatten())
print("*" * 50)
print("val_ccc (pearson):{} ({})".format(ccc_result[0], ccc_result[1]))
print("val_ccc_tricks (pearson):{} ({})".format(ccc_result_tricks[0], ccc_result_tricks[1]))

plt.figure(figsize=(30, 10))
plt.plot(Y_val)
plt.plot(preds)
plt.plot(preds_tricks)
plt.show()