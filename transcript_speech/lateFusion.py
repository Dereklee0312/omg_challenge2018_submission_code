import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, concatenate, Flatten
from keras.models import load_model
from scipy.stats import pearsonr
from scipy.signal import butter, lfilter
import os
import tensorflow as tf

# Custom Attention Layer (from previous)
from models.attlayer import AttentionWeightedAverage

# Parameters from text model (new_lstm.py)
subjects = [1,2,3,4,5,6,7,8,9,10]
stories_val = [2]
stories_train = [1,4,5,8]
base_path = "./vectors/val2/"
labels_path = "./data/original_dataset/annotations/"
normalize_labels = True
train_min_y = -1.0  # Replace with computed if different
train_max_y = 1.0
smooth = 0
window_size = 100
stride = 50
embedding_size = 11
lstm_output_dim = 64
subject_vector_size = 2
initial_dropout = 0.2
final_dropout = 0.2
second_last_dim = 32
lstm_attention = True
activation = "relu"

# Parameters from speech model (new_rnn.py)
slice_sec = 8.0
overlap_percent = 0.2
stft_hop_sec = 0.01  # 10ms
valence_fs = 25
stft_fs = 1 / stft_hop_sec
stride_sec = slice_sec * (1 - overlap_percent)
slice_stft = int(slice_sec * stft_fs)
stride_stft = int(stride_sec * stft_fs)
slice_val = int(slice_sec * valence_fs)
stride_val = int(stride_sec * valence_fs)

# Path to speech training predictors for global mean/std (replace with actual path)
TRAINING_PREDICTORS = "./speech_predictors/training_2A_S_predictors.npy"  # Update this path

# Load global mean/std for speech
training_predictors = np.load(TRAINING_PREDICTORS)
tr_mean = np.mean(training_predictors)
tr_std = np.std(training_predictors)

# Data loading functions
def get_X(story, subject, modality):
    file_name = "/Subject_" + str(subject) + "_Story_" + str(story) + "_aligned.npy"
    latent_vecs_path = base_path + modality + "_aligned" + file_name
    try:
        X = np.load(latent_vecs_path)
        return X
    except FileNotFoundError:
        print(f"Missing file: {latent_vecs_path}")
        return None

def get_Y(story, subject, smooth=0):
    file_name = "/Subject_" + str(subject) + "_Story_" + str(story) + ".csv"
    labels_path_full = labels_path + file_name
    try:
        Y = open(labels_path_full).read().split("\n")[1:-1]
        Y = [float(x) for x in Y]
        if smooth > 0:
            Y = butter_lowpass_filter_bidirectional(np.array(Y), cutoff=smooth, fs=25, order=1)
        return np.array(Y)
    except FileNotFoundError:
        print(f"Missing file: {labels_path_full}")
        return None

# Butterworth filter (if smooth > 0)
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
    y_first_pass = butter_lowpass_filter(data[::-1], cutoff, fs, order)
    y_second_pass = butter_lowpass_filter(y_first_pass[::-1], cutoff, fs, order)
    return y_second_pass

# CCC function
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

def batch_CCC(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Means over sequence dimension (axis=1), shape: (batch_size,)
    mean_true = tf.reduce_mean(y_true, axis=1)
    mean_pred = tf.reduce_mean(y_pred, axis=1)
    
    # Center the data
    centered_true = y_true - tf.expand_dims(mean_true, 1)
    centered_pred = y_pred - tf.expand_dims(mean_pred, 1)
    
    # Covariance over sequence, shape: (batch_size,)
    covar = tf.reduce_mean(centered_true * centered_pred, axis=1)
    
    # Standard deviations, shape: (batch_size,)
    std_true = tf.sqrt(tf.reduce_mean(tf.square(centered_true), axis=1) + 1e-8)
    std_pred = tf.sqrt(tf.reduce_mean(tf.square(centered_pred), axis=1) + 1e-8)
    
    # CCC per sequence
    numerator = 2.0 * covar
    denominator = tf.square(std_true) + tf.square(std_pred) + tf.square(mean_true - mean_pred) + 1e-8
    ccc = numerator / denominator
    
    # Average CCC over batch and compute loss
    mean_ccc = tf.reduce_mean(ccc)
    loss = 1.0 - mean_ccc
    return loss

# f_trick
def f_trick(Y_train, preds):
    Y_train_flat = Y_train.flatten()
    preds_flat = preds.flatten()
    s0 = np.std(Y_train_flat)
    m1 = np.mean(preds_flat)
    s1 = np.std(preds_flat)
    m0 = np.mean(Y_train_flat)
    norm_preds = s0 * (preds_flat - m1) / (s1 + 1e-10) + m0
    return norm_preds

# Build text model
def build_text_model():
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

# Load models
text_model = build_text_model()
text_model.load_weights("./models/tmp_weights.h5")

speech_model = load_model("./models/bigru_PROVA2.keras", custom_objects={"batch_CCC": batch_CCC})

# Load training labels for f_trick (frame-level)
Y_frame_list_train = []
for subject in subjects:
    for story in stories_train:
        Y = get_Y(story, subject)
        if Y is not None:
            Y_frame_list_train.append(Y)
Y_concat_train = np.concatenate(Y_frame_list_train)

# Evaluate on validation
ccc_text_list = []
ccc_speech_list = []
ccc_fused_list = []

for subject in subjects:
    story = stories_val[0]  # 2
    print(f"Processing Subject {subject}, Story {story}...")
    
    Y = get_Y(story, subject)
    if Y is None:
        continue
    num_frames = len(Y)
    
    # Text predictions
    X_text = get_X(story, subject, "text")
    if X_text is None or len(X_text) < window_size:
        print(f"Skipping text for Subject {subject}, Story {story}")
        continue
    
    num_windows = ((num_frames - window_size) // stride) + 1
    X_windowed_text = []
    subjects_windowed = []
    for i in range(num_windows):
        start_i = i * stride
        end_i = start_i + window_size
        X_windowed_text.append(X_text[start_i:end_i])
        subjects_windowed.append(subject - 1)
    
    X_new_text = np.array(X_windowed_text)
    X_new_late_subject = np.array(subjects_windowed)
    
    predictions_text = text_model.predict([X_new_late_subject, X_new_text]).flatten()
    if normalize_labels:
        predictions_text = predictions_text * (train_max_y - train_min_y) + train_min_y
    
    # Reconstruct per-frame for text
    pred_text_frame = np.zeros(num_frames)
    count_text = np.zeros(num_frames)
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        pred_text_frame[start:end] += predictions_text[i]
        count_text[start:end] += 1
    pred_text_frame /= np.maximum(count_text, 1)
    
    # Speech (audio) predictions
    X_speech = get_X(story, subject, "audio")  # Assumes "audio" modality path exists
    if X_speech is None or len(X_speech) < slice_stft:
        print(f"Skipping speech for Subject {subject}, Story {story}")
        continue
    
    num_segments = ((len(X_speech) - slice_stft) // stride_stft) + 1
    X_windowed_speech = []
    for i in range(num_segments):
        start = i * stride_stft
        end = start + slice_stft
        win = X_speech[start:end]
        win = (win - np.mean(win)) / (np.std(win) + 1e-10)  # Per-slice normalize
        win = (win - tr_mean) / (tr_std + 1e-10)  # Global normalize
        X_windowed_speech.append(win)
    
    X_new_speech = np.array(X_windowed_speech)
    predictions_speech = speech_model.predict(X_new_speech)
    
    # Denormalize speech predictions (from [0,1] to [-1,1])
    predictions_speech = predictions_speech * 2 - 1
    
    # Reconstruct per-frame for speech
    pred_speech_frame = np.zeros(num_frames)
    count_speech = np.zeros(num_frames)
    for i in range(num_segments):
        start = i * stride_val
        end = start + slice_val
        pred_speech_frame[start:end] += predictions_speech[i]
        count_speech[start:end] += 1
    pred_speech_frame /= np.maximum(count_speech, 1)
    
    # Apply f_trick to both (using frame-level training labels)
    pred_text_frame = f_trick(Y_concat_train, pred_text_frame)
    pred_speech_frame = f_trick(Y_concat_train, pred_speech_frame)
    
    # Late fusion: average
    fused_frame = (pred_text_frame + pred_speech_frame) / 2
    
    # Compute CCC
    ccc_text, _ = ccc(Y, pred_text_frame)
    ccc_speech, _ = ccc(Y, pred_speech_frame)
    ccc_fused, _ = ccc(Y, fused_frame)
    
    ccc_text_list.append(ccc_text)
    ccc_speech_list.append(ccc_speech)
    ccc_fused_list.append(ccc_fused)
    
    print(f"Subject {subject}, Story {story}: CCC Text = {ccc_text:.4f}, CCC Speech = {ccc_speech:.4f}, CCC Fused = {ccc_fused:.4f}")

# Average over all validation videos
mean_ccc_text = np.mean(ccc_text_list)
mean_ccc_speech = np.mean(ccc_speech_list)
mean_ccc_fused = np.mean(ccc_fused_list)

print("\nAverage CCC on Validation:")
print(f"Text: {mean_ccc_text:.4f}")
print(f"Speech: {mean_ccc_speech:.4f}")
print(f"Fused: {mean_ccc_fused:.4f}")