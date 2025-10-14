import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, concatenate, Flatten
from scipy.signal import butter, lfilter
from models.attlayer import AttentionWeightedAverage  # Ensure this is accessible
import os

# Parameters (must match training settings)
subjects = [1,2,3,4,5,6,7,8,9,10]
stories_new = [1,4,5,8]
base_path = "./data/"
output_path = "./predictions/"  # Directory to save prediction .npy files
window_size = 100
stride = 50
embedding_size = 11
normalize_labels = True  # Match training setting
smooth = 0  # Match training setting
lstm_output_dim = 64
subject_vector_size = 2
initial_dropout = 0.2
final_dropout = 0.2
second_last_dim = 32
lstm_attention = True
activation = "relu"

# Normalization parameters from training (replace with actual values)
# These should be saved during training or computed from training labels
train_min_y = 0.0  # Replace with actual min_y from training
train_max_y = 1.0  # Replace with actual max_y from training

# Data loading function from original script
def get_X(story, subject, modality):
    file_name = "/Subject_" + str(subject) + "_Story_" + str(story) + "_aligned.npy"
    base_path = "./vectors/val2/"
    latent_vecs_path = base_path + "text_aligned" + file_name
    try:
        X = np.load(latent_vecs_path)
        return X
    except FileNotFoundError:
        print(f"Missing file: {latent_vecs_path}")
        return None  # Return None to skip invalid files

# Butterworth filter functions (if smooth > 0, included for completeness)
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

# Model architecture
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

# Post-processing (if used during training)
def f_trick(Y_train, preds):
    Y_train_flat = Y_train.flatten()
    preds_flat = preds.flatten()
    s0 = np.std(Y_train_flat)
    m1 = np.mean(preds_flat)
    s1 = np.std(preds_flat)
    m0 = np.mean(Y_train_flat)
    norm_preds = s0 * (preds_flat - m1) / (s1 + 1e-10) + m0
    return norm_preds

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Load model and weights
print("Loading model...")
model = build_model()
model.load_weights("tmp_weights.h5")  # Load saved weights

# Process each subject-story pair
print("Generating predictions...")
for subject in subjects:
    for story in stories_new:
        print(f"Processing Subject {subject}, Story {story}...")
        
        # Load and preprocess data
        X = get_X(story, subject, "text")
        if X is None:
            print(f"Skipping Subject {subject}, Story {story} due to missing data")
            continue
        
        num_frames = len(X)
        if num_frames < window_size:
            print(f"Skipping Subject {subject}, Story {story}: Not enough frames ({num_frames} < {window_size})")
            continue
        
        # Window the data
        X_windowed = []
        subjects_windowed = []
        num_windows = ((num_frames - window_size) // stride) + 1
        for i in range(num_windows):
            start_i = i * stride
            end_i = start_i + window_size
            X_win = X[start_i:end_i]
            X_windowed.append(X_win)
            subjects_windowed.append(subject - 1)  # Adjust subject ID to 0-based index
        
        X_new = np.array(X_windowed)  # Shape: [num_windows, window_size, embedding_size]
        X_new_late_subject = np.array(subjects_windowed)  # Shape: [num_windows]
        
        # Sanity check
        print(f"Subject {subject}, Story {story} - Input shape: {X_new.shape}, Subject IDs: {len(X_new_late_subject)}")
        
        # Make predictions
        predictions = model.predict([X_new_late_subject, X_new], batch_size=500)  # Shape: [num_windows, 1]
        
        # Reverse normalization (if applied during training)
        if normalize_labels:
            predictions = predictions * (train_max_y - train_min_y) + train_min_y
        
        # Optional: Apply f_trick (requires Y_train)
        # Y_train = ... # Load training labels if needed
        # predictions = f_trick(Y_train, predictions)
        
        # Flatten predictions
        predictions = predictions.flatten()  # Shape: [num_windows]
        
        # Save predictions to .npy file
        output_file = os.path.join(output_path, f"Subject_{subject}_Story_{story}_predictions.npy")
        np.save(output_file, predictions)
        print(f"Saved predictions to {output_file} (Shape: {predictions.shape})")

print("Prediction generation complete.")