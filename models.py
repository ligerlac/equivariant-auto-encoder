from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer,
    Input,
    Reshape,
    Flatten,
    Activation,
    AveragePooling2D,
    Conv2D,
    Dense,
    Dropout,
    UpSampling2D,
    Conv2DTranspose
)


class TrivialModel:

    @staticmethod
    def get_model():
        inputs = Input((28, 28, 1))
        return Model(inputs, inputs, name='trivial-model')


class BaselineAutoEncoder:
    
    @staticmethod
    def get_model():
        inputs = Input((28, 28, 1))
        x = Reshape((28, 28, 1))(inputs)
        x = Conv2D(10, (3, 3), strides=1, padding="same")(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((2, 2))(x)
        x = Conv2D(15, (3, 3), strides=1, padding="same")(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(60, activation="relu")(x)
        x = Dense(14 * 14 * 30)(x)
        x = Reshape((14, 14, 30))(x)
        x = Activation("relu")(x)
        x = Conv2D(15, (3, 3), strides=1, padding="same")(x)
        x = Activation("relu")(x)
        # x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(10, (3, 3), strides=2, padding="same")(x)
        x = Conv2D(10, (3, 3), strides=1, padding="same")(x)
        x = Activation("relu")(x)
        outputs = Conv2D(1, (3, 3), activation="relu", strides=1, padding="same")(x)
        return Model(inputs, outputs, name="baseline-auto-encoder")


class BruteForceMultiplied:
    """
    create an equivariant model with brute force:
    duplicate input by applying all elements of the respective symmetry group
    feed each of these transformed inputs into the baseline model
    combine all results pixels-wise (sum v mean v max ...)
    """
    pass