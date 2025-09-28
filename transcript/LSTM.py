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
import keras.backend as K
from keras.layers import TimeDistributed
from keras.utils import plot_model
from models.attlayer import AttentionWeightedAverage
import time

# Parameters
# subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
subjects = [1]
# subjects = [1]
stories_train = [1]  # [1,2,4,5,8]
stories_val = [1]
normalize_labels = True
smooth = 0  # smooth labels
modalities = ["text"]
base_path = "./data/"
labels_path = "./data/original_dataset/annotations/"
checkpoint_filename = "tmp_weights.h5"
# General params
batch_size = 500
epochs = 1000
patience = 5
lr = 0.0001
embedding_size = 11
window_size = 200
# window_size = 100
initial_dropout = 0.2  # % of lexicon features to drop
# Model params
subject_vector_early = False
subject_vector_late = True
subject_vecotr_size = 2
lstm_stateful = False
lstm_units = 64
lstm_dropout = 0.2
initial_dropout = 0.2
activation = "relu"
lstm_output_dim = 64
lstm_attention = True
final_dropout = 0.2
lstm_attention_type = "softmax"
second_last_dim = 32
# Others
day_time = time.strftime("%Y-%m-%d_%H_%M_%S")

# Prepare subject indices
subjects_train = []
for i in range(len(subjects) * len(stories_train)):
    subjects_train.append(i // len(stories_train))
subjects_val = []
for i in range(len(subjects) * len(stories_val)):
    subjects_val.append(i // len(stories_val))

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
        # Y = pd.read_csv(labels_path_full, header=None, skiprows=1, dtype=float).values.flatten()
        # if smooth > 0:
        #     Y = butter_lowpass_filter_bidirectional(np.array(Y), cutoff=smooth, fs=25, order=1)
        # if normalize_labels:
        Y = open(labels_path_full).read().split("\n")[1:]
        Y = [float(x) for x in Y]
        #     Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y) + 1e-10)
        return Y
    except FileNotFoundError:
        print(f"Missing file: {labels_path_full}")
        return np.zeros(100)

def get_all_X(stories, subjects, modalities):
    X_dic = {}
    for modality in modalities:
        print("Loading modality ", modality)
        X_list = []
        for subject in subjects:
            for story in stories:
                X = get_X(story, subject, modality)
                X_list.append(X)
        X_dic[modality] = np.concatenate(X_list, axis=0)
    return X_dic['text']  # Return text modality array

def get_all_Y(stories, subjects, normalize_labels=False, smooth=0):
    Y_list = []
    for subject in subjects:
        for story in stories:
            Y = get_Y(story, subject)
            Y_list.append(Y)
            if smooth > 0:
                Y = butter_lowpass_filter_bidirectional(np.array(Y), cutoff=smooth, fs=25, order=1)
            if normalize_labels:
                Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y) + 1e-10)
    return np.concatenate(Y_list, axis=0)

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
    return 1-ccc

# model
def build_model():
    input_late_subject = Input(shape=(1,),dtype='int32')
    input_lstm = Input(shape=(window_size, embedding_size))
    seq_input_drop = Dropout(initial_dropout)(input_lstm)
    # Sequence modeling network
    if lstm_attention:
        lstm_output = LSTM(lstm_output_dim, return_sequences=True)(seq_input_drop)
        lstm_output, _ = AttentionWeightedAverage(name='attlayer')(lstm_output)
    else:
        lstm_output = LSTM(lstm_output_dim, return_sequences=False)(seq_input_drop)
    #lstm_output_drop = Dropout(final_dropout)(lstm_output)
   
    subject_embedding = Embedding(11, subject_vecotr_size)(input_late_subject)
    subject_embedding = Flatten()(subject_embedding)
    subject_concat = concatenate([subject_embedding, lstm_output])
    second_last = Dense(second_last_dim, name="second_last", activation=activation)(subject_concat)
    second_last = Dropout(final_dropout)(second_last)
    outputs = Dense(1)(second_last)
   
    return Model(inputs=[input_late_subject, input_lstm], outputs=outputs)

# measure ccc when epoch ends
class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._data = []
    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(model.predict(X_val))
   
        ccc_result, rho_result = ccc(y_val, y_predict)
       
        self._data.append({
           'ccc': ccc_result,
           'rho': rho_result
        })
        print("ccc = %f, pearson=%f" % (ccc_result[0], rho_result[0]) )
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

# Training
print("-- Loading training data --")
X_train = get_all_X(stories_train, subjects, modalities)
Y_train = get_all_Y(stories_train, subjects, normalize_labels, smooth)

# Validation
print("-- Loading validation data --")
X_val = get_all_X(stories_val, subjects, modalities)
Y_val = get_all_Y(stories_val, subjects, normalize_labels, smooth)

# Convert subject indices to numpy arrays
X_train_late_subject = np.array(subjects_train)
X_val_late_subject = np.array(subjects_val)

# Sanity checks
print("train lexicon vectors:", X_train.shape)
print("train subject late:", len(X_train_late_subject))
print("train labels:", len(Y_train))
print("val lexicon vectors:", X_val.shape)
print("val subject late:", len(X_val_late_subject))
print("val labels:", len(Y_val))

# Training
opt = optimizers.Adam(learning_rate=lr, decay=0.0)
model = build_model()
model.compile(loss=ccc_error,
              optimizer=opt)
metrics = Metrics()
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