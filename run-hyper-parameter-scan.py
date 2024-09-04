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


def main(args):

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_shirts, test_shirts = train_images[train_labels==0], test_images[test_labels==0]
    outliers = test_images[test_labels!=0]

    model = BaselineAutoEncoder

    tuner = keras_tuner.RandomSearch(
        model.build_tunable_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        project_name=args.output,
        overwrite=False,
    )

    tuner.search_space_summary()

    tuner.search(
        x=train_shirts,
        y=train_shirts,
        epochs=args.epochs,
        validation_data=(test_shirts, test_shirts),
        verbose=args.verbose
    )

    tuner.results_summary()


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
