"""Utilities for Interactive Medical Image Segmentation."""
import tensorflow as tf
from libtiff import TIFF as tiff
import os
import numpy as np

CLASSES = 9


def get_images(path):
    """Load dataset images from the given path."""
    images = [
        "rotated/" + tif
        for tif in os.listdir(path + "sat/rotated")
        if tif[-4:] == ".tif"
    ]

    orig = []
    seg = []
    for img in images:
        tif = tiff.open(path + "sat/" + img)
        orig.append(tif.read_image())
        tif.close()
        tif = tiff.open(path + "gt/" + img)
        seg.append(tif.read_image())
        tif.close()

    return np.array(orig), np.array(seg)


def get_datasets(path, val_split, test_split, batch_size):
    """Get training, validation and test datasets.

    Two iterators are created: "train" and "test". The "train" iterator is
    made from the training dataset and repeats indefinitely, while the "test"
    iterator can be initialized with the validation and the test datasets using
    the init ops "val" and "test".

    Args:
        path(str): The path to the dataset
        val_split(float): The fraction of validation split
        test_split(float): The fraction of test split
        batch_size(int): The batch size

    Returns:
        dict: {
            "iterators": {
                "train": `tf.data.Iterator`, "test": `tf.data.Iterator`
            },
            "init_ops": {"val": `tf.Operation`, "test": `tf.Operation`}
        }

    """
    # Sanity checks
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if val_split < 0 or val_split > 1:
        raise ValueError(
            "Invalid value for validation split (given: {})".format(val_split)
        )
    if test_split < 0 or test_split > 1:
        raise ValueError(
            "Invalid value for test split (given: {})".format(test_split)
        )
    if (val_split + test_split) >= 1:
        raise ValueError(
            "No split for training data (remaining: {})".format(
                1 - (val_split + test_split)
            )
        )

    def one_hot(x, y):
        return tf.to_float(x), tf.one_hot(tf.squeeze(y, axis=-1), CLASSES)

    orig, seg = get_images(path)
    dataset = tf.data.Dataset.from_tensor_slices((orig, seg))
    dataset = dataset.map(one_hot).shuffle(1000)

    val_length = int(len(orig) * val_split)
    test_length = int(len(orig) * test_split)

    val_dataset = dataset.take(val_length)
    test_dataset = dataset.skip(val_length).take(test_length)
    train_dataset = dataset.skip(val_length + test_length)

    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.batch(
        batch_size, drop_remainder=True
    ).repeat()

    train_iterator = train_dataset.make_one_shot_iterator()
    test_iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types, train_dataset.output_shapes
    )
    val_init_op = test_iterator.make_initializer(val_dataset)
    test_init_op = test_iterator.make_initializer(test_dataset)
    del orig, seg

    return {
        "iterators": {"train": train_iterator, "test": test_iterator},
        "init_ops": {"val": val_init_op, "test": test_init_op},
    }


class EarlyStopper(object):
    """An object to do early stopping on a loss/metric.

    Status can be polled using EarlyStopper().stop
    """

    def __init__(self, steps, diff, mode="reduce"):
        """Initialize stopper with data.

        Args:
            steps(int): No. of steps after which to stop.
            diff(float): The minimum change to test in early stopping.
            mode(str): Whether the loss/metric should reduce or increase.
        """
        mode = mode.lower().strip()
        if mode not in ["reduce", "increase"]:
            raise ValueError("Mode should be one of 'reduce' or 'increase'")
        else:
            self.mode = mode
        if mode == "reduce":
            self._value = 1e7
        else:
            self._value = -1e7

        if steps < 0:
            raise ValueError("Steps for early stopping must be non-negative")
        self._init_steps = steps
        self._steps = steps

        self._diff = diff
        self.stop = False

    def update(self, value):
        """Update stopper with details and set status.

        Args:
            value(float): The current value of the loss/metric.
        """
        diff = self._value - value
        if (self.mode == "reduce" and diff < self._diff) or (
            self.mode == "increase" and diff > self._diff
        ):
            self._steps -= 1
            if self._steps == 0:
                self.stop = True
                return
        else:
            self._steps = self._init_steps
            self._value = value
        self.stop = False
