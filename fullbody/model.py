from resnet3d import Resnet3DBuilder

from keras.layers import Dense
from keras.models import Model


def create_reg_resnet18_3D(img_x, img_y, ch_n, seq_len, tgt_size):
    resnet18_3D = Resnet3DBuilder.build_resnet_18(
        (seq_len, img_x, img_y, ch_n), tgt_size
    )
    output = resnet18_3D.get_layer("flatten_1").output
    output = Dense(32, activation="relu")(output)
    output = Dense(1, activation="linear")(output)
    reg_resnet18_3D = Model(resnet18_3D.input, output)

    return reg_resnet18_3D
