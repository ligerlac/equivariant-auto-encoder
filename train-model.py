import argparse

from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist

from utils import IsValidFile, CreateFolder
from models import TrivialModel


def main(args):

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    model = TrivialModel.get_model()
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    mc = ModelCheckpoint(f"{args.output}/model.keras", save_best_only=True)
    log = CSVLogger(f"{args.output}/training.log", append=False)

    model.fit(
        x=train_images,
        y=train_images,
        epochs=args.epochs,
        validation_data=(test_images, test_images),
        callbacks=[mc, log],
        verbose=args.verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", "-o",
        action=CreateFolder,
        type=Path,
        default="models/",
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
