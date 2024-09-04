import argparse
import glob
import re
import keras_tuner

import pandas as pd
import numpy as np

from pathlib import Path
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model

from utils import IsValidFile, IsReadableDir, CreateFolder
from drawing import Draw
from models import BaselineAutoEncoder


def main(args):

    draw = Draw(output_dir=args.output, interactive=args.interactive)

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_shirts, test_shirts = train_images[train_labels==0], test_images[test_labels==0]
    outliers = test_images[test_labels!=0]

    model = BaselineAutoEncoder

    tuner = keras_tuner.RandomSearch(
        model.build_tunable_model,
        max_trials=0,
        project_name=args.input,
        overwrite=False,
    )

    tuner.results_summary()
    best_model = tuner.get_best_models()[0]


    # model_dict, log_dict = {}, {}

    # for path in glob.glob(f'{args.input}/*.log'):
    #     log = pd.read_csv(path)
    #     path = path.split('/')[-1]
    #     ls = int(re.search(r"training-(\d+)\.log", path).group(1))
    #     log_dict[ls] = log

    # for path in glob.glob(f'{args.input}/*.keras'):
    #     model = load_model(path)
    #     path = path.split('/')[-1]
    #     ls = int(re.search(r"model-(\d+)\.keras", path).group(1))
    #     model_dict[ls] = log

    # latent_sizes, best_val_losses, outlier_losses, loss_ratios = [], [], [], []
    # for ls, log in sorted(log_dict.items())[5:]:
    #     latent_sizes.append(ls)
    #     idx = np.argmin(log['val_loss'])
    #     best_val_losses.append(log['val_loss'][idx])
    #     outlier_losses.append(log['outlier_loss'][idx])
    #     loss_ratios.append(log['val_loss'][idx] / log['outlier_loss'][idx])
    
    # draw.plot_val_loss_vs_latent_size(latent_sizes, best_val_losses, name='best-val-loss-vs-latent-size')
    # draw.plot_loss_vs_latent_size(latent_sizes, best_val_losses, outlier_losses, name='loss-vs-latent-size')
    # draw.plot_loss_ratios(latent_sizes, loss_ratios, name='loss-ratios')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i",
        action=IsReadableDir,
        type=Path,
        default="models/hpo",
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
