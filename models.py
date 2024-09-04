from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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

from keras_tuner import HyperParameters


class TrivialModel:

    @staticmethod
    def get_model():
        inputs = Input((28, 28, 1))
        return Model(inputs, inputs, name='trivial-model')


class BaselineAutoEncoder:
    
    @staticmethod
    def get_model(
        latent_size=50, n_outer_conv=10, outer_kernel=3, n_inner_conv=15, inner_kernel=3
    ):
        inputs = Input((28, 28, 1))
        x = Reshape((28, 28, 1))(inputs)
        x = Conv2D(n_outer_conv, (outer_kernel, outer_kernel), strides=1, padding="same")(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((2, 2))(x)
        x = Conv2D(n_inner_conv, (inner_kernel, inner_kernel), strides=1, padding="same")(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(latent_size, activation="relu")(x)
        x = Dense(14 * 14 * n_inner_conv)(x)
        x = Reshape((14, 14, n_inner_conv))(x)
        x = Activation("relu")(x)
        x = Conv2D(n_inner_conv, (inner_kernel, inner_kernel), strides=1, padding="same")(x)
        x = Activation("relu")(x)
        # x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(n_outer_conv, (outer_kernel, outer_kernel), strides=2, padding="same")(x)
        x = Conv2D(n_outer_conv, (outer_kernel, outer_kernel), strides=1, padding="same")(x)
        x = Activation("relu")(x)
        outputs = Conv2D(1, (3, 3), activation="relu", strides=1, padding="same")(x)
        return Model(inputs, outputs, name="baseline-auto-encoder")

    @staticmethod
    def build_tunable_model(hp: HyperParameters):
        ls = hp.Int("latent_size", min_value=5, max_value=100, default=50)
        n_o_c = hp.Int("n_outer_conv", min_value=3, max_value=30, default=10)
        o_k = hp.Int("outer_kernel", min_value=2, max_value=6, default=3)
        n_i_c = hp.Int("n_inner_conv", min_value=3, max_value=30, default=15)
        i_k = hp.Int("inner_kernel", min_value=2, max_value=6, default=3)
        model = BaselineAutoEncoder.get_model(
            latent_size=ls, n_outer_conv=n_o_c, outer_kernel=o_k, n_inner_conv=n_i_c, inner_kernel=i_k
        )
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        return model


class BruteForceMultiplied:
    """
    create an equivariant model with brute force:
    duplicate input by applying all elements of the respective symmetry group
    feed each of these transformed inputs into the baseline model
    combine all results pixels-wise (sum v mean v max ...)
    """

    @staticmethod
    def get_model(latent_size=60):
        model_ = BaselineAutoEncoder.get_model()
        inputs = Input((28, 28, 1))
        x1 = inputs
        x2 = inputs[:, ::-1, :]
        y1 = model_(x1)
        y2 = model_(x2)
        outputs = y1 + y2
        return Model(inputs, outputs, name='bruteforce-multiplied')
