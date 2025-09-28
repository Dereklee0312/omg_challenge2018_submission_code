import numpy as np
import pandas as pd
import os

# Assume you have per-file frame counts from CSVs (e.g., load len(gt_valence))
# Directories from your pipeline
lex_dir = "../data/text/lexicons_features/"  # CSVs with per-word features
annotation_dir = "data/original_dataset/annotations/"  # For frame counts
output_dir = "vectors/val2/text_aligned/"  # Aligned .npy
os.makedirs(output_dir, exist_ok=True)

def upsample_to_frames(lex_csv, ann_csv):
    # Load per-word features (num_words x 11)
    df_lex = pd.read_csv(lex_csv, header=None)
    features = df_lex.to_numpy()  # Shape: (num_words, 11)
    
    # Load frame count from labels
    gt_valence = pd.read_csv(ann_csv, header=None).values.flatten()
    num_frames = len(gt_valence)
    
    # Recompute word frame durations (simplified; use your time_to_frame logic)
    # Assume even distribution for demo; in practice, load from TSV or SRT
    num_words = len(features)
    frames_per_word = num_frames // num_words  # Approx; adjust for exact
    remain = num_frames % num_words
    
    # Create frame-level array
    aligned_features = np.zeros((num_frames, 11))
    frame_idx = 0
    for i in range(num_words):
        word_frames = frames_per_word + (1 if i < remain else 0)
        aligned_features[frame_idx:frame_idx + word_frames] = features[i]
        frame_idx += word_frames
    
    # Pad remaining frames (silence) with neutral (e.g., zeros or mean)
    if frame_idx < num_frames:
        aligned_features[frame_idx:] = np.mean(features, axis=0)  # Or 0.5 for valence-like
    
    return aligned_features

for su in range(1, 11):
    for st in range(1, 6):
        lex_csv = f"{lex_dir}/Subject_{su}_Story_{st}_lex.csv"
        ann_csv = f"{annotation_dir}/Subject_{su}_Story_{st}.csv"
        if os.path.exists(lex_csv) and os.path.exists(ann_csv):
            aligned = upsample_to_frames(lex_csv, ann_csv)
            np.save(f"{output_dir}/Subject_{su}_Story_{st}_aligned.npy", aligned)
            print(f"Aligned {aligned.shape} for Subject_{su}_Story_{st}")