#!/usr/bin/env python3
# coding: utf-8
"""Interactive Medical Image Segmentation for Eye-in-the-Sky."""
from __future__ import print_function
from __future__ import division
import tensorflow as tf
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import get_datasets, EarlyStopper
from model import Model

parser = ArgumentParser(
    description="Interactive Medical Image Segmentation for Eye-in-the-Sky",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--data-path", type=str, default="./", help="path to dataset"
)

# Known hyperparams
parser.add_argument("--batch-size", type=int, default=32, help="batch size")
parser.add_argument(
    "--weight-decay", type=float, default=5e-4, help="L2 regularisation scale"
)
parser.add_argument(
    "--max-steps", type=int, default=1000, help="maximum no. of training steps"
)

# Unknown hyperparams
parser.add_argument("--random-seed", type=int, default=5, help="random seed")
parser.add_argument(
    "--dropout", type=float, default=0.5, help="dropout probability"
)
parser.add_argument(
    "--val-split",
    type=float,
    default=0.2,
    help="fraction of data to use as validation",
)
parser.add_argument(
    "--test-split",
    type=float,
    default=0.2,
    help="fraction of data to use as test data",
)
parser.add_argument(
    "--early-stop-diff",
    type=float,
    default=5e-4,
    help="minimum change for early stopping",
)
parser.add_argument(
    "--early-stop-steps",
    type=int,
    default=5,
    help="minimum steps to wait for early stopping (0 disables early "
    "stopping)",
)

# Miscellaneous
parser.add_argument(
    "--log-steps",
    type=int,
    default=10,
    help="steps after which to test on validation",
)
parser.add_argument(
    "--log-dir",
    type=str,
    default="./logdir",
    help="where to store Tensorboard summaries",
)

args = parser.parse_args()

tf.set_random_seed(args.random_seed)

print("Loading dataset...")
data = get_datasets(
    args.data_path, args.val_split, args.test_split, args.batch_size
)
print("Dataset loaded")

print("Building graph...")
model = Model(data, tf.train.AdamOptimizer(), args.weight_decay, args.dropout)
print("Graph built")


# Limit GPU usage
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    print("Starting training...")
    if args.early_stop_steps != 0:
        stopper = EarlyStopper(
            args.early_stop_steps, args.early_stop_diff, "reduce"
        )
    else:
        stopper = None

    try:
        model.train(
            sess,
            args.max_steps,
            args.log_dir,
            args.log_steps,
            stopper,
            stop_on="loss",
        )
    except KeyboardInterrupt:
        pass
    print("Training done")

    print("Starting testing...")
    model.evaluate("test")
