from raw_face_model import conv_3d_id_model
# from utils import *
from utils import ccc_error, create_img_dataset, light_id_generator, make_id_vector

import keras.backend as K
from keras import optimizers

import keras.callbacks as cb

import numpy as np
import tensorflow as tf
import time
import os

# save_path = "/OMG_empathy_challange/models/raw_face/trained/"
# img_path = "/OMG_Empathy2019/full_body/full_body/Subject_{0}_Story_{1}/Subject_img/"
# lbl_path = "/OMG_Empathy2019/labels/Subject_{0}_Story_{1}.csv"

day_time = time.strftime("%Y-%m-%d_%H_%M_%S")

save_path = "./model/"
os.makedirs(save_path, exist_ok=True)
# img_path = "/OMG_Empathy2019/full_body/full_body/Subject_{0}_Story_{1}/Subject_img/"
# lbl_path = "/OMG_Empathy2019/labels/Subject_{0}_Story_{1}.csv"

img_path = "../landmarks/faces_extracted/training/Subject_{0}_Story_{1}/Subject_face/"
lbl_path = "../data/Training/Annotations/Subject_{0}_Story_{1}.csv"


seq_len = 10
img_x = 48
img_y = 48
ch_n = 1

id_len = 10

down_sampling = 1

batch_size = 128
epochs = 100


##### Creating model

K.clear_session()

m = conv_3d_id_model(seq_len, img_x, img_y, ch_n, id_len).create()
opti = optimizers.Adam(learning_rate=0.0001)

# m.compile(loss='mean_squared_error', optimizer=opti) ## need to add ccc metric
m.compile(loss=ccc_error, optimizer=opti)  ## need to add ccc metric


##### load training and make generator

sbj_n_s = range(1, 11)
str_n_s = [2, 4, 5, 8]

lbl_tr = np.concatenate(
    [
        np.loadtxt(lbl_path.format(sbj_n, str_n), skiprows=1)[::down_sampling]
        for str_n in str_n_s
        for sbj_n in sbj_n_s
    ]
).reshape(-1, 1)
img_tr = create_img_dataset(
    lbl_tr.shape[0], img_x, img_y, ch_n, str_n_s, sbj_n_s, down_sampling
)
ids_tr = make_id_vector(str_n_s, sbj_n_s, lbl_path)

print("train images loaded with shape: ", img_tr.shape)
print("train labels loaded with shape: ", lbl_tr.shape)

lw_gen_tr = light_id_generator(img_tr[:], lbl_tr[:], ids_tr[:], seq_len, batch_size)
steps_per_epoch_tr = lw_gen_tr.stp_per_epoch

# Create tf.data.Dataset with explicit output signatures for Keras 3
train_dataset = tf.data.Dataset.from_generator(
    lw_gen_tr.generate,
    output_signature=(
        {
            "main_input": tf.TensorSpec(shape=(batch_size, seq_len, img_x, img_y, ch_n), dtype=tf.float64),
            "aux_input": tf.TensorSpec(shape=(batch_size, id_len), dtype=tf.float64),
        },
        tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float64),
    ),
)


##### load validation and make generator

sbj_n_s = range(1, 11)
str_n_s = [2]

lbl_vl = np.concatenate(
    [
        np.loadtxt(lbl_path.format(sbj_n, str_n), skiprows=1)[::down_sampling]
        for str_n in str_n_s
        for sbj_n in sbj_n_s
    ]
).reshape(-1, 1)
img_vl = create_img_dataset(
    lbl_vl.shape[0], img_x, img_y, ch_n, str_n_s, sbj_n_s, down_sampling
)
ids_vl = make_id_vector(str_n_s, sbj_n_s, lbl_path)

print("valid images loaded with shape: ", img_vl.shape)
print("valid labels loaded with shape: ", lbl_vl.shape)

lw_gen_vl = light_id_generator(img_vl, lbl_vl, ids_vl, seq_len, batch_size)
steps_per_epoch_vl = lw_gen_vl.stp_per_epoch

# Create tf.data.Dataset with explicit output signatures for Keras 3
valid_dataset = tf.data.Dataset.from_generator(
    lw_gen_vl.generate,
    output_signature=(
        {
            "main_input": tf.TensorSpec(shape=(batch_size, seq_len, img_x, img_y, ch_n), dtype=tf.float64),
            "aux_input": tf.TensorSpec(shape=(batch_size, id_len), dtype=tf.float64),
        },
        tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float64),
    ),
)


##### Setting callbacks

# save_name = save_path+'conv_3D_global_regression_ID_{0}_.{epoch:02d}-{val_loss:.2f}.h5'.format(time.strftime("%Y-%m-%d_%H:%M"))
save_name = save_path + "/conv_3D_raw_face.{epoch:02d}-{val_loss:.2f}.keras"

bckup_callback = cb.ModelCheckpoint(
    # save_name,
    # monitor="val_loss",
    # verbose=0,
    # save_best_only=True,
    # save_weights_only=False,
    # mode="auto",
    # period=1,
    save_name,
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    save_freq='epoch',
)


stop_callback = cb.EarlyStopping(monitor="val_loss", patience=5)


callbacks_list = [
    bckup_callback,
    stop_callback,
    cb.TensorBoard(log_dir="logs/" + day_time),
]


m.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch_tr,
    epochs=epochs,
    callbacks=callbacks_list,
    validation_data=valid_dataset,
    validation_steps=steps_per_epoch_vl,
)


# # Plot predictions
# m.load_weights(checkpoint_filename)
# preds = m.predict_generator(lw_gen_vl.generate())
# preds_norm = norm_pred(lbl_tr, preds)
# ccc_result = ccc(lbl_vl, preds.flatten())
# ccc_result_norm = ccc(lbl_vl, preds_norm.flatten())
# print("*"*50)
# print("val_ccc (pearson):{} ({})".format(ccc_result[0], ccc_result[1]))
# print("val_ccc_tricks (pearson):{} ({})".format(ccc_result_norm[0], ccc_result_norm[1]))

# plt.figure(figsize=(30, 10))
# plt.plot(lbl_vl)
# plt.plot(preds)
# plt.plot(preds_norm)
