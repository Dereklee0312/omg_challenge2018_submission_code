# CONVOLUTIONAL NEURAL NETWORK
# tuned as in https://www.researchgate.net/publication/306187492_Deep_Convolutional_Neural_Networks_and_Data_Augmentation_for_Environmental_Sound_Classification

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Flatten, LSTM, TimeDistributed, Reshape, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
import utilities_func as uf
import loadconfig
import configparser
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
np.random.seed(1)

# Load dataset...
config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)
SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_cfg_path(value: str) -> str:
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((SCRIPT_DIR / p).resolve())

# Load parameters from config file
NEW_CONV_MODEL = cfg.get('model', 'save_model')
TRAINING_PREDICTORS = _resolve_cfg_path(cfg.get('model', 'training_predictors_load'))
TRAINING_TARGET = _resolve_cfg_path(cfg.get('model', 'training_target_load'))
VALIDATION_PREDICTORS = _resolve_cfg_path(cfg.get('model', 'validation_predictors_load'))
VALIDATION_TARGET = _resolve_cfg_path(cfg.get('model', 'validation_target_load'))
NEW_CONV_MODEL = _resolve_cfg_path(NEW_CONV_MODEL)
SEQ_LENGTH = cfg.getint('preprocessing', 'sequence_length')
print("Training predictors: " + TRAINING_PREDICTORS)
print("Training target: " + TRAINING_TARGET)
print("Validation predictors: " + VALIDATION_PREDICTORS)
print("Validation target: " + VALIDATION_TARGET)

# Load datasets
training_predictors = np.load(TRAINING_PREDICTORS)
training_target = np.load(TRAINING_TARGET)
validation_predictors = np.load(VALIDATION_PREDICTORS)
validation_target = np.load(VALIDATION_TARGET)

# Rescale datasets to mean 0 and std 1 (validation with respect to training mean and std)
tr_mean = np.mean(training_predictors)
tr_std = np.std(training_predictors)
v_mean = np.mean(validation_predictors)
v_std = np.std(validation_predictors)
training_predictors = np.subtract(training_predictors, tr_mean)
training_predictors = np.divide(training_predictors, tr_std)
validation_predictors = np.subtract(validation_predictors, tr_mean)
validation_predictors = np.divide(validation_predictors, tr_std)

# Normalize target between 0 and 1
training_target = np.multiply(training_target, 0.5)
training_target = np.add(training_target, 0.5)
validation_target = np.multiply(validation_target, 0.5)
validation_target = np.add(validation_target, 0.5)

# Hyperparameters
batch_size = 32
num_epochs = 200
lstm1_depth = 250
hidden_size = 8
drop_prob = 0.3
dense_size = 100
regularization_lambda = 0.01

reg = regularizers.l2(regularization_lambda)
sgd = optimizers.SGD(learning_rate=0.001, momentum=0.5)
opt = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Custom loss (vectorized for TF 2.x compatibility)
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

time_dim = training_predictors.shape[1]
features_dim = training_predictors.shape[2]

# Callbacks
best_model = ModelCheckpoint(NEW_CONV_MODEL, monitor='val_loss', save_best_only=True, mode='min')  # Save the best model
early_stopping_monitor = EarlyStopping(patience=5)  # Stop training when the model is not improving
callbacks_list = [early_stopping_monitor, best_model]

# Model definition
input_data = Input(shape=(time_dim, features_dim))
gru = Bidirectional(GRU(lstm1_depth, return_sequences=True))(input_data)
norm = BatchNormalization()(gru)
hidden = TimeDistributed(Dense(hidden_size, activation='linear'))(norm)
drop = Dropout(drop_prob)(hidden)
flat = Flatten(name='flatten')(drop)
out = Dense(SEQ_LENGTH, activation='linear')(flat)

# Model creation
valence_model = Model(inputs=input_data, outputs=out)
valence_model.compile(loss=batch_CCC, optimizer=opt)

print(valence_model.summary())

# Model training
history = valence_model.fit(training_predictors, training_target, epochs=num_epochs, validation_data=(validation_predictors, validation_target), callbacks=callbacks_list, batch_size=batch_size, shuffle=True)

print("Train loss = " + str(min(history.history['loss'])))
print("Validation loss = " + str(min(history.history['val_loss'])))

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MODEL PERFORMANCE', size=15)
plt.ylabel('loss', size=15)
plt.xlabel('Epoch', size=15)
plt.xticks(size=15)
plt.yticks(size=15)
plt.legend(['train', 'validation'], fontsize=12)

plt.show()
