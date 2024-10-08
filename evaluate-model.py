import argparse

import pandas as pd

from pathlib import Path
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model

from utils import IsValidFile, IsReadableDir, CreateFolder
from drawing import Draw


def main(args):

    draw = Draw(output_dir=args.output, interactive=args.interactive)

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_shirts, test_shirts = train_images[train_labels==0], test_images[test_labels==0]

    model = load_model(f"{args.input}/model.keras")

    log = pd.read_csv(f"{args.input}/training.log")
    draw.plot_loss_history(log, f"training-history-{model.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i",
        action=IsReadableDir,
        type=Path,
        default="models/",
        help="Path to directory w/ trained models",
    )
    parser.add_argument(
        "--output", "-o",
        action=CreateFolder,
        type=Path,
        default="plots/",
        help="Path to directory where plots will be stored",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively display plots as they are created",
        default=False,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Output verbosity",
        default=False,
    )
    main(parser.parse_args())
