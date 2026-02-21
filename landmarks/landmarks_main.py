from model import *
from utils import *

from pathlib import Path


import keras
from keras import optimizers
from keras.callbacks import CSVLogger

from contextlib import redirect_stdout

import time
# from time import time

subject_data = True
actor_data = False


if subject_data == actor_data:
    raise Exception("Choose between subject and actor data")

window_size = 5
(
    X_training,
    Y_training,
    X_validation,
    Y_validation,
    indexes_training,
    indexes_validation,
) = load_dataset(window_size)

day_time = time.strftime("%Y-%m-%d_%H_%M_%S")
experiment_name = "experiment_" + day_time
BASE_DIR = Path(__file__).resolve().parent
experiments_dir = BASE_DIR / "experiments"
experiments_dir.mkdir(parents=True, exist_ok=True)
experiment_dir = experiments_dir / experiment_name
experiment_dir.mkdir(parents=True, exist_ok=False)

filepath = str(experiment_dir) + "/"


batch_size = 512
epochs = 1000
patience = 1000
lr = 0.000001

opt = optimizers.Adam(learning_rate=lr)

N_features = X_training.shape[2]

model = build_model(window_size, N_features)
# model = build_model_LSTM()

print(model.summary())

model.compile(
    loss=ccc_error,  # ccc_error #mean_squared_error #pearson_error
    optimizer=opt,
)

# metrics_callback = Metrics()
# csv_logger = CSVLogger("training.log")
metrics_callback = Metrics(X_validation, Y_validation, filepath, batch_size)
csv_logger = CSVLogger("training.log")

callbacks_list = [  # csv_logger,
    metrics_callback,
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience),
    keras.callbacks.TensorBoard(log_dir=str(BASE_DIR / "logs" / experiment_name)),
    # keras.callbacks.ModelCheckpoint(filepath=experiment_name+'/weights_epoch_{epoch:02d}.h5', monitor='val_loss', save_best_only=False)
]

with open(experiment_dir / "modelsummary.txt", "w") as f:
    with redirect_stdout(f):
        model.summary()

# pp= 8193
# ppp = 3000

history = model.fit(
    X_training,
    Y_training,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_validation, Y_validation),
    callbacks=callbacks_list,
)

# metrics_callback.get_data()

save_latent_training = False
save_predictions_training = False
save_latent_test = False
save_predictions_test = True

model.load_weights(str(experiment_dir / "best_model.h5"))
save_predictions(model, Y_training)
