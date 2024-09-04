import argparse
import keras_tuner

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist

from utils import IsValidFile, CreateFolder, CustomLoss
from models import TrivialModel, BaselineAutoEncoder


def build_tunable_model(hp: keras_tuner.HyperParameters):
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


def main(args):

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_shirts, test_shirts = train_images[train_labels==0], test_images[test_labels==0]
    outliers = test_images[test_labels!=0]

    tuner = keras_tuner.RandomSearch(
        build_tunable_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=2,
        directory=args.output,
        overwrite=True,
    )

    mc = ModelCheckpoint(f"{args.output}/model.keras", save_best_only=True)
    oll = CustomLoss(inputs=outliers, name='outlier_loss')
    log = CSVLogger(f"{args.output}/training.log", append=False)
    
    tuner.search(
        x=train_shirts,
        y=train_shirts,
        epochs=args.epochs,
        validation_data=(test_shirts, test_shirts),
        callbacks=[mc, oll, log],
        verbose=args.verbose
    )

    tuner.search_space_summary()
    best_model = tuner.get_best_models()[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", "-o",
        action=CreateFolder,
        type=Path,
        default="models/hpo",
        help="Path to directory where models will be stored",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        help="Number of training epochs",
        default=10,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Output verbosity",
        default=False,
    )
    main(parser.parse_args())
