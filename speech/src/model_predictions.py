import numpy as np
from tensorflow.keras.models import load_model
from scipy.signal import butter, lfilter
import os
import loadconfig
import configparser
import tensorflow as tf

# Custom loss function (required for loading the model)
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

# Butterworth filter functions
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

# Post-processing function
def f_trick(Y_train, preds):
    Y_train_flat = Y_train.flatten()
    preds_flat = preds.flatten()
    s0 = np.std(Y_train_flat)
    m1 = np.mean(preds_flat)
    s1 = np.std(preds_flat)
    m0 = np.mean(Y_train_flat)
    norm_preds = s0 * (preds_flat - m1) / (s1 + 1e-10) + m0
    return norm_preds

# Parameters (adapted from model_predictions.py)
subjects = [1,2,3,4,5,6,7,8,9,10]
stories_new = [1,4,5,8]
base_path = "./vectors/val2/"
modality = "audio_aligned"  # Assuming this is the folder for audio features; change if different
output_path = "./speech_predictions/"  # Directory to save prediction .npy files
normalize_labels = True
smooth = 0  # Set to >0 to enable smoothing (e.g., 1)
use_f_trick = False  # Set to True to enable f_trick post-processing
train_min_y = -1.0
train_max_y = 1.0  # Not directly used; reverse is 2*x - 1 since normalized to 0-1

# Load config to get SEQ_LENGTH and TRAINING_PREDICTORS
config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)
SEQ_LENGTH = cfg.getint('preprocessing', 'sequence_length')
TRAINING_PREDICTORS = cfg.get('model', 'training_predictors_load')
TRAINING_TARGET = cfg.get('model', 'training_target_load')

# Compute training mean and std
training_predictors = np.load(TRAINING_PREDICTORS)
tr_mean = np.mean(training_predictors)
tr_std = np.std(training_predictors)

# Load training targets for f_trick if enabled (loaded as original -1 to 1)
if use_f_trick:
    training_target = np.load(TRAINING_TARGET)

# Load the saved model
print("Loading model...")
custom_objects = {'batch_CCC': batch_CCC}
model = load_model("../models/bigru_PROVA2.keras", custom_objects=custom_objects)

# Get input shape from model
time_dim = model.input_shape[1]
features_dim = model.input_shape[2]

# Overlap fraction from PDF (20%)
overlap_frac = 0.2
stride = int(time_dim * (1 - overlap_frac))
valence_stride = int(SEQ_LENGTH * (1 - overlap_frac))

# Data loading function (adapted)
def get_X(story, subject, modality):
    file_name = "/Subject_" + str(subject) + "_Story_" + str(story) + "_aligned.npy"
    latent_vecs_path = base_path + modality + file_name
    try:
        X = np.load(latent_vecs_path)
        return X
    except FileNotFoundError:
        print(f"Missing file: {latent_vecs_path}")
        return None

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Process each subject-story pair
print("Generating predictions...")
for subject in subjects:
    for story in stories_new:
        print(f"Processing Subject {subject}, Story {story}...")
        
        # Load and preprocess data
        X = get_X(story, subject, modality)
        if X is None:
            print(f"Skipping Subject {subject}, Story {story} due to missing data")
            continue
        
        num_frames = len(X)
        if num_frames < time_dim:
            print(f"Skipping Subject {subject}, Story {story}: Not enough frames ({num_frames} < {time_dim})")
            continue
        
        # Window the data
        num_windows = ((num_frames - time_dim) // stride) + 1
        X_windowed = []
        for i in range(num_windows):
            start_i = i * stride
            end_i = start_i + time_dim
            X_win = X[start_i:end_i]
            # Normalize using training mean/std
            X_win = (X_win - tr_mean) / tr_std
            X_windowed.append(X_win)
        
        X_new = np.array(X_windowed)  # Shape: [num_windows, time_dim, features_dim]
        
        # Sanity check
        print(f"Subject {subject}, Story {story} - Input shape: {X_new.shape}")
        
        # Make predictions
        predictions = model.predict(X_new, batch_size=500)  # Shape: [num_windows, SEQ_LENGTH]
        
        # Combine overlapping predictions
        full_val_length = ((num_windows - 1) * valence_stride) + SEQ_LENGTH
        full_preds = np.zeros(full_val_length)
        counts = np.zeros(full_val_length)
        for i in range(num_windows):
            start = i * valence_stride
            end = start + SEQ_LENGTH
            full_preds[start:end] += predictions[i]
            counts[start:end] += 1
        full_preds /= (counts + 1e-8)
        
        # Reverse normalization (from 0-1 to -1-1)
        if normalize_labels:
            full_preds = 2 * full_preds - 1
        
        # Optional: Apply smoothing
        if smooth > 0:
            fs = 25  # Valence sampling rate (1 / 0.04s = 25 Hz)
            cutoff = 0.1
            order = 1
            full_preds = butter_lowpass_filter_bidirectional(full_preds, cutoff, fs, order)
        
        # Optional: Apply f_trick
        if use_f_trick:
            full_preds = f_trick(training_target, full_preds)
        
        # Save predictions to .npy file
        output_file = os.path.join(output_path, f"Subject_{subject}_Story_{story}_predictions.npy")
        np.save(output_file, full_preds)
        print(f"Saved predictions to {output_file} (Shape: {full_preds.shape})")

print("Prediction generation complete.")