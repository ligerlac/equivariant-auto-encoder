from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer,
    Input
)


class TrivialModel:
    def get_model():
        inputs = Input((28, 28, 1))
        return Model(inputs, inputs, name='trivial-model')
