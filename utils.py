import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class CustomLoss(Callback):
    def __init__(self, inputs, name='custom_loss'):
        super().__init__()
        self.inputs = inputs
        self.name = name

    def on_epoch_end(self, epoch, log):
        loss_fn = tf.keras.losses.get(self.model.loss)
        preds = self.model.predict(self.inputs, verbose=0)
        loss = loss_fn(self.inputs.flatten(), preds.flatten()).numpy()
        log[self.name] = loss


class IsReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid path".format(prospective_dir)
            )
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError(
                "{0} is not a readable directory".format(prospective_dir)
            )


class IsValidFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_file = values
        if not os.path.exists(prospective_file):
            raise argparse.ArgumentTypeError(
                "{0} is not a valid file".format(prospective_file)
            )
        else:
            setattr(namespace, self.dest, prospective_file)


class CreateFolder(argparse.Action):
    """
    Custom action: create a new folder if not exist. If the folder
    already exists, do nothing.

    The action will strip off trailing slashes from the folder's name.
    """

    def create_folder(self, folder_name):
        """
        Create a new directory if not exist. The action might throw
        OSError, along with other kinds of exception
        """
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        # folder_name = folder_name.rstrip(os.sep)
        folder_name = os.path.normpath(folder_name)
        return folder_name

    def __call__(self, parser, namespace, values, option_string=None):
        if type(values) == list:
            folders = list(map(self.create_folder, values))
        else:
            folders = self.create_folder(values)
        setattr(namespace, self.dest, folders)


def predict_single_image(model, image):
    image = np.expand_dims(image, axis=(0, -1))
    pred = model.predict(image, verbose=False)
    return np.squeeze(pred, axis=(0, -1))
