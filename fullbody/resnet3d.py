from __future__ import annotations

from keras import layers, models


class Resnet3DBuilder:
    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return Resnet3DBuilder._build_resnet(
            input_shape,
            num_outputs,
            block_counts=[2, 2, 2, 2],
        )

    @staticmethod
    def _build_resnet(input_shape, num_outputs, block_counts):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv3D(
            64, kernel_size=7, strides=2, padding="same", use_bias=False
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling3D(pool_size=3, strides=2, padding="same")(x)

        filters = 64
        for stage_index, blocks in enumerate(block_counts):
            for block_index in range(blocks):
                stride = 1
                if stage_index > 0 and block_index == 0:
                    stride = 2
                x = Resnet3DBuilder._basic_block(
                    x, filters=filters, stride=stride
                )
            filters *= 2

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Flatten(name="flatten_1")(x)
        outputs = layers.Dense(num_outputs, activation="linear")(x)
        return models.Model(inputs=inputs, outputs=outputs, name="resnet3d_18")

    @staticmethod
    def _basic_block(x, filters, stride):
        shortcut = x
        x = layers.Conv3D(
            filters, kernel_size=3, strides=stride, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(
            filters, kernel_size=3, strides=1, padding="same", use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)

        if shortcut.shape[-1] != filters or stride != 1:
            shortcut = layers.Conv3D(
                filters, kernel_size=1, strides=stride, padding="same", use_bias=False
            )(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x
