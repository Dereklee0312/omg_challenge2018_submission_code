"""Evaluate speech model and export predictions in legacy and canonical formats."""
import numpy as np
import os
import pandas
import essentia.standard as ess
from tensorflow.keras.models import load_model, Model
from scipy.signal import filtfilt, butter
import utilities_func as uf
from calculateCCC import ccc2
import feat_analysis2 as fa
import tensorflow as tf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from shared_utils.config_loader import load_defaults, resolve_manifest
from shared_utils.split_validation import split_for_story
from shared_utils.pred_schema import ensure_prediction_dir, write_prediction_csv
from speech_config import speech_paths, speech_preprocessing, speech_sampling, speech_stft

defaults = load_defaults()
manifest = resolve_manifest(defaults)
paths_cfg = speech_paths()
pre_cfg = speech_preprocessing()
sampling_cfg = speech_sampling()
stft_cfg = speech_stft()

# Get values from shared defaults-backed speech config.
EVALUATION_PREDICTORS_LOAD = paths_cfg["evaluation_predictors_load"]
REFERENCE_PREDICTORS_LOAD = paths_cfg["reference_predictors_load"]
EVALUATION_TARGET_LOAD = paths_cfg["evaluation_target_load"]
LLD_DIR = paths_cfg["last_latent_dim_dir"]
SEQ_LENGTH = int(pre_cfg["sequence_length"])
MODEL = paths_cfg["load_model"]
SR = int(sampling_cfg["sr"])
HOP_SIZE = int(stft_cfg["hop_size"])
MODEL_OUTPUT_FOLDER = paths_cfg["model_output_folder"]
os.makedirs(MODEL_OUTPUT_FOLDER, exist_ok=True)
csv_out_dir = ensure_prediction_dir(defaults["predictions"]["base_dir"], "speech")

fps = 25  # Annotations per second
hop_annotation = SR / fps
frames_per_annotation = hop_annotation / float(HOP_SIZE)
feats_per_frame = 8
feats_per_valence = int(frames_per_annotation * feats_per_frame)

# Custom loss function (vectorized for TF 2.x compatibility; batch_size not used directly)
def batch_CCC(y_true, y_pred):
    """Compute batch CCC loss used when loading the trained model."""
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

# Load classification model and create latent extractor sub-model
valence_model = load_model(MODEL, custom_objects={'CCC': uf.CCC, 'batch_CCC': batch_CCC})
latent_model = Model(inputs=valence_model.input, outputs=valence_model.get_layer('flatten').output)  # Assuming flatten layer is named 'flatten'

# Load datasets rescaling
reference_predictors = np.load(REFERENCE_PREDICTORS_LOAD)
ref_mean = np.mean(reference_predictors)
ref_std = np.std(reference_predictors)
predictors = np.load(EVALUATION_PREDICTORS_LOAD)
target = np.load(EVALUATION_TARGET_LOAD)

print("")
print("using model: " + MODEL)

def predict_datapoint(input_sound, input_annotation):
    """Predict valence sequence for one sample and persist outputs."""
    sr, samples = uf.wavread(input_sound)  # load
    e_samples = uf.preemphasis(samples, sr)  # pre-emphasis with sr
    predictors = fa.extract_features(e_samples)  # extract features
    predictors = np.subtract(predictors, ref_mean)
    predictors = np.divide(predictors, ref_std)
    target = pandas.read_csv(input_annotation)
    target = target.values
    target = np.reshape(target, (target.shape[0]))
    final_pred = []
    start = 0
    while start < (len(target) - SEQ_LENGTH):
        start_features = int(start * frames_per_annotation)
        stop_features = int((start + SEQ_LENGTH) * frames_per_annotation)
        predictors_temp = predictors[start_features:stop_features]
        predictors_temp = predictors_temp.reshape(1, predictors_temp.shape[0], predictors_temp.shape[1])
        prediction = valence_model.predict(predictors_temp)
        final_pred.extend(prediction[0])
        perc = int(float(start) / (len(target) - SEQ_LENGTH) * 100)
        print("Computing prediction: " + str(perc) + "%")
        start += SEQ_LENGTH
    # Compute prediction for last frame
    predictors_temp = predictors[-int(SEQ_LENGTH * frames_per_annotation):]
    predictors_temp = predictors_temp.reshape(1, predictors_temp.shape[0], predictors_temp.shape[1])
    prediction = valence_model.predict(predictors_temp)
    missing_samples = len(target) - len(final_pred)
    final_pred.extend(prediction[0][-missing_samples:])
    final_pred = np.array(final_pred)
    
    # POSTPROCESSING
    # Normalize between -1 and 1
    final_pred = np.multiply(final_pred, 2.)
    final_pred = np.subtract(final_pred, 1.)
    
    # Apply f_trick
    ann_folder = defaults["paths"]["train_annotations"]
    target_mean, target_std = uf.find_mean_std(ann_folder)
    final_pred = uf.f_trick(final_pred, target_mean, target_std)
    
    # Apply butterworth filter
    b, a = butter(3, 0.01, 'low')
    final_pred = filtfilt(b, a, final_pred)
    
    # Save legacy CSV output and canonical split-tagged CSV output.
    name = os.path.basename(input_sound).replace(".mp4.wav", "")
    subject = name.split("_")[1]  # Extract subject number
    story = name.split("_")[3]    # Extract story number
    output_file = os.path.join(MODEL_OUTPUT_FOLDER, f"Subject_{subject}_Story_{story}.csv")
    df = pandas.DataFrame({"valence": final_pred})
    df.to_csv(output_file, index=False)
    print(f"Saved prediction to {output_file}")
    split = split_for_story(int(story), manifest["stories_train"], manifest["stories_val"])
    write_prediction_csv(
        csv_out_dir,
        int(subject),
        int(story),
        frame_idx=list(range(len(final_pred))),
        y_pred=final_pred.tolist(),
        y_true=target.tolist(),
        manifest_id=manifest["manifest_id"],
        split=split,
    )
    
    ccc = ccc2(final_pred, target)
    print("CCC = " + str(ccc))
    return ccc

def extract_LLD_datapoint(input_sound, input_annotation):
    """Extract latent representation sequence for one audio/annotation pair."""
    sr, samples = uf.wavread(input_sound)  # load
    e_samples = uf.preemphasis(samples, sr)  # pre-emphasis with sr
    predictors = fa.extract_features(e_samples)  # extract features
    predictors = np.subtract(predictors, ref_mean)
    predictors = np.divide(predictors, ref_std)
    target = pandas.read_csv(input_annotation)
    target = target.values
    target = np.reshape(target, (target.shape[0]))
    start = 0
    final_vec = np.array([])
    # Compute last latent dim for each frame
    while start < (len(target) - SEQ_LENGTH):
        start_features = int(start * frames_per_annotation)
        stop_features = int((start + SEQ_LENGTH) * frames_per_annotation)
        predictors_temp = predictors[start_features:stop_features]
        predictors_temp = predictors_temp.reshape(1, predictors_temp.shape[0], predictors_temp.shape[1])
        features_temp = latent_model.predict(predictors_temp)
        features_temp = np.reshape(features_temp, (SEQ_LENGTH, feats_per_valence))
        if final_vec.shape[0] == 0:
            final_vec = features_temp
        else:
            final_vec = np.concatenate((final_vec, features_temp), axis=0)
        print('Progress: ' + str(int(100 * (final_vec.shape[0] / float(len(target))))) + '%')
        start += SEQ_LENGTH
    # Compute last latent dim for last frame
    predictors_temp = predictors[-int(SEQ_LENGTH * frames_per_annotation):]
    predictors_temp = predictors_temp.reshape(1, predictors_temp.shape[0], predictors_temp.shape[1])
    features_temp = latent_model.predict(predictors_temp)
    features_temp = np.reshape(features_temp, (SEQ_LENGTH, feats_per_valence))
    missing_samples = len(target) - final_vec.shape[0]
    last_vec = features_temp[-missing_samples:]
    final_vec = np.concatenate((final_vec, last_vec), axis=0)
    return final_vec

def evaluate_all_data(sound_dir, annotation_dir):
    """Run prediction+CCC evaluation across manifest validation stories."""
    file_list = os.listdir(annotation_dir)
    file_list = file_list[:]  # Copy list
    file_list = [
        f for f in file_list if any(f"_Story_{s}.csv" in f for s in manifest["stories_val"])
    ]
    ccc_values = []
    for datapoint in file_list:
        annotation_file = annotation_dir + '/' + datapoint
        name = datapoint.split('.')[0]
        print('Processing: ' + name)
        sound_file = sound_dir + '/' + name + ".mp4.wav"
        temp_ccc = predict_datapoint(sound_file, annotation_file)
        ccc_values.append(temp_ccc)
    ccc_values = np.array(ccc_values)
    mean_ccc = np.mean(ccc_values)
    min_ccc = np.min(ccc_values)
    max_ccc = np.max(ccc_values)

    print("Mean CCC = " + str(mean_ccc))
    print("Min CCC = " + str(min_ccc))
    print("Max CCC = " + str(max_ccc))

def extract_LLD_dataset(sound_dir, annotation_dir):
    """Extract and save latent representations for all annotation files."""
    file_list = os.listdir(annotation_dir)
    file_list = file_list[:]  # Copy list
    for datapoint in file_list:
        annotation_file = annotation_dir + '/' + datapoint
        name = datapoint.split('.')[0]
        print('Processing: ' + name)
        sound_file = sound_dir + '/' + name + ".mp4.wav"
        lld = extract_LLD_datapoint(sound_file, annotation_file)
        output_filename = LLD_DIR + '/' + name + '.npy'
        np.save(output_filename, lld)

if __name__ == "__main__":
    sound_dir = defaults["paths"]["val_audio"]
    annotation_dir = defaults["paths"]["val_annotations"]
    evaluate_all_data(sound_dir, annotation_dir)
