# models/attlayer.py
from tensorflow.keras import Layer
import tensorflow.keras.backend as K

class AttentionWeightedAverage(Layer):
    def __init__(self, name='attlayer', attention_type='softmax', **kwargs):
        super(AttentionWeightedAverage, self).__init__(name=name, **kwargs)
        self.attention_type = attention_type

    def call(self, x):
        # x: LSTM output (batch, timesteps, units)
        if self.attention_type == 'softmax':
            weights = K.softmax(K.sum(x, axis=-1))  # Sum over units, softmax over timesteps
        else:
            weights = K.mean(x, axis=-1)  # Fallback: mean pooling
        weighted_avg = K.sum(x * K.expand_dims(weights, -1), axis=1)  # Weighted sum
        return weighted_avg, weights

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]