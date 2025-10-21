from keras.models import load_model
import keras.backend as K
import numpy as np
import os
import cv2
import tensorflow as tf

# Define paths
model_path = './models/conv_3D_raw_face_2025-09-25_04-35-30.keras'
base_img_path = './data/extracted_faces'
output_path = './rawface_predictions/'
os.makedirs(output_path, exist_ok=True)

# Parameters from training script
seq_len = 10
img_x = 48
img_y = 48
ch_n = 1  # Grayscale
id_len = 10
stride = 1  # Dense predictions to match frame count as closely as possible

# Custom loss function (required for loading the model)
def ccc_error(y_true, y_pred):
    """Concordance Correlation Coefficient loss"""
    x = tf.cast(y_true, tf.float32)
    y = tf.cast(y_pred, tf.float32)

    mx = tf.reduce_mean(x)
    my = tf.reduce_mean(y)

    xm = x - mx
    ym = y - my

    r_num = tf.reduce_sum(xm * ym)
    r_den = tf.sqrt(tf.reduce_sum(tf.square(xm)) * tf.reduce_sum(tf.square(ym)))
    r = r_num / (r_den + 1e-8)

    ccc = 2 * r * tf.math.reduce_std(x) * tf.math.reduce_std(y) / (
        tf.math.reduce_variance(x) + tf.math.reduce_variance(y) + tf.square(mx - my) + 1e-8
    )

    return 1 - ccc

# Load the model with custom loss
print("Loading model...")
model = load_model(model_path, custom_objects={'ccc_error': ccc_error})
model.summary()  # Optional: Print model summary

# Load images function (adapted from training script)
def load_images_from_directory(base_path, video_name, max_images=None):
    possible_paths = [
        os.path.join(base_path, 'Training', video_name, 'Subject_img'),
        os.path.join(base_path, 'Validation', video_name, 'Subject_img'),
        os.path.join(base_path, 'Testing', video_name, 'Subject_img'),
        os.path.join(base_path, 'Faces', video_name, 'Subject_img'),
        os.path.join(base_path, video_name, 'Subject_img')
    ]
    
    subject_img_path = next((path for path in possible_paths if os.path.exists(path)), None)
    if subject_img_path is None:
        print(f"Warning: No Subject_img directory found for {video_name}")
        return np.array([])
    
    image_files = [f for f in os.listdir(subject_img_path) if f.endswith('.png')]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    if max_images:
        image_files = image_files[:max_images]
    
    images = []
    for img_file in image_files:
        img_path = os.path.join(subject_img_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if img.shape != (48, 48):
                img = cv2.resize(img, (48, 48))
            img = img.astype(np.float32) / 255.0
            images.append(img)
    
    return np.array(images)

# Subjects and stories matching model_predictions.py
subjects = range(1, 11)  # 1 to 10
stories = [1, 4, 5, 8]

# Generate predictions
print("Generating predictions...")
for subject in subjects:
    for story in stories:
        print(f"Processing Subject {subject}, Story {story}...")
        
        video_name = f"Subject_{subject}_Story_{story}"
        images = load_images_from_directory(base_img_path, video_name)
        num_images = len(images)
        
        if num_images < seq_len:
            print(f"Skipping Subject {subject}, Story {story}: Not enough frames ({num_images} < {seq_len})")
            continue
        
        # Create subject ID vector
        id_vector = np.zeros(id_len, dtype=np.float32)
        if subject - 1 < id_len:
            id_vector[subject - 1] = 1.0
        
        # Generate sliding windows and predict
        num_preds = num_images - seq_len + 1
        predictions = []
        
        for i in range(0, num_preds, stride):
            img_seq = images[i:i + seq_len]
            img_seq = img_seq.reshape(1, seq_len, img_x, img_y, ch_n)
            id_vec = id_vector.reshape(1, id_len)
            
            pred = model.predict({'main_input': img_seq, 'aux_input': id_vec}, verbose=0)
            predictions.append(pred[0][0])  # Extract scalar prediction
        
        predictions = np.array(predictions)
        
        # Pad to match the total number of images (repeat first/last prediction for edge cases)
        if len(predictions) < num_images:
            pad_length = num_images - len(predictions)
            padding = np.full(pad_length, predictions[-1] if predictions.size > 0 else 0.0)
            predictions = np.concatenate([predictions, padding])
        elif len(predictions) > num_images:
            predictions = predictions[:num_images]  # Truncate if overshooting
        
        # Save predictions
        output_file = os.path.join(output_path, f"Subject_{subject}_Story_{story}_predictions.npy")
        np.save(output_file, predictions)
        print(f"Saved predictions to {output_file} (Shape: {predictions.shape})")

print("Prediction generation complete.")