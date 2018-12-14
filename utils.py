"""Utilities for Interactive Medical Image Segmentation."""
import tensorflow as tf
from libtiff import TIFF as tiff
import os
import numpy as np
from dataset_rot import EXCLUDE, get_colours

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


def get_test_images(path):
    """Load actual test dataset images."""
    images = [
        tif for tif in os.listdir(path + "sat_test") if tif[-4:] == ".tif"
    ]

    for img in images:
        tif = tiff.open(path + "sat_test/" + img)
        image = tif.read_image()
        tif.close()
        yield image, img


def one_hot(x, y):
    """One hot the outputs, and make input float."""
    return tf.to_float(x), tf.one_hot(tf.squeeze(y, axis=-1), CLASSES)


def get_datasets(path, val_split, test_split, batch_size):
    """Get the test dataset from excluded images.

    This function is similar to `get_old_datasets`, but it uses the excluded
    images as both validation and test datasets.

    Args:
        path(str): The path to the dataset
        val_split(float): Ignored; kept for compatibility
        test_split(float): Ignored; kept for compatibility
        batch_size(int): The batch size

    Returns:
        dict: {
            "iterators": {
                "train": `tf.data.Iterator`, "test": `tf.data.Iterator`
            },
            "init_ops": {"val": `tf.Operation`, "test": `tf.Operation`},
            "path": str
        }

    """
    if path[-1] != "/":
        path += "/"
    info = get_old_datasets(path, 0, 0, batch_size)
    colours = get_colours(path + "gt/")

    def excl_gen():
        images = EXCLUDE
        for img in images:
            tif = tiff.open(path + "sat/" + img)
            sat = tif.read_image()
            tif.close()
            tif = tiff.open(path + "gt/" + img)
            gt = tif.read_image()
            tif.close()

            new_gt = [colours.index(tuple(i)) for i in gt.reshape((-1, 3))]
            gt = np.reshape(new_gt, (*(gt.shape[:2]), 1)).astype(np.uint8)
            yield sat, gt

    dataset = tf.data.Dataset.from_generator(
        excl_gen,
        (tf.uint16, tf.uint8),
        (tf.TensorShape([None, None, 4]), tf.TensorShape([None, None, 1])),
    )
    dataset = dataset.map(one_hot).batch(1)
    test_iterator = tf.data.Iterator.from_structure(
        dataset.output_types, dataset.output_shapes
    )
    test_init_op = test_iterator.make_initializer(dataset)

    return {
        "iterators": {
            "train": info["iterators"]["train"],
            "test": test_iterator,
            "actual": info["iterators"]["actual"],
        },
        "init_ops": {"val": test_init_op, "test": test_init_op},
        "path": path,
    }


def get_old_datasets(path, val_split, test_split, batch_size):
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
            "init_ops": {"val": `tf.Operation`, "test": `tf.Operation`},
            "path": str
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

    if path[-1] != "/":
        path += "/"
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

    # Get actual test dataset
    act_dataset = tf.data.Dataset.from_generator(
        lambda: get_test_images(path),
        (tf.uint16, tf.string),
        (tf.TensorShape([None, None, 4]), tf.TensorShape([])),
    )
    act_dataset = act_dataset.map(lambda x, y: (tf.to_float(x), y)).batch(1)
    act_iterator = act_dataset.make_one_shot_iterator()

    return {
        "iterators": {
            "train": train_iterator,
            "test": test_iterator,
            "actual": act_iterator,
        },
        "init_ops": {"val": val_init_op, "test": test_init_op},
        "path": path,
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
