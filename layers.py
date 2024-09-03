import tensorflow as tf

from tensorflow.keras.layers import (
    Layer,
    Activation,
    AveragePooling2D,
    MaxPooling2D,
    Conv2D,
    ZeroPadding2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Reshape,
    UpSampling2D,
    BatchNormalization,
)


class CircularPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(CircularPadding2D, self).__init__(**kwargs)

    def call(self, inputs):
        padding_height, padding_width = self.padding
        if padding_width > 0:
            inputs = tf.concat([inputs[:, :, -padding_width:], inputs, inputs[:, :, :padding_width]], axis=2)
        if padding_height > 0:
            inputs = tf.concat([inputs[:, -padding_height:, :], inputs, inputs[:, :padding_height, :]], axis=1)
        return inputs


class OneSidedCircularPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(OneSidedCircularPadding2D, self).__init__(**kwargs)

    def call(self, inputs):
        padding_height, padding_width = self.padding
        if padding_width > 0:
            inputs = tf.concat([inputs[:, :, -padding_width:], inputs], axis=2)
        if padding_height > 0:
            inputs = tf.concat([inputs[:, -padding_height:, :], inputs], axis=1)
        return inputs


class CirculantDense(Layer):
    def __init__(self, **kwargs):
        super(CirculantDense, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-2])
        self.spectrum = self.add_weight(
            shape=(input_dim,),
            initializer='random_normal',
            trainable=True,
            name='spectrum'
        )
        
    def call(self, inputs):
        circulant_matrix = LinearOperatorCirculant(self.spectrum, input_output_dtype=tf.dtypes.float32)
        # output = tf.matmul(inputs, circulant_matrix)
        output = tf.matmul(circulant_matrix, inputs)
        return output

    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0])
