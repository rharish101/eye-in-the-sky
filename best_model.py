#!/usr/bin/env python3
"""Best model for Interactive Medical Image Segmentation."""
from __future__ import print_function
from __future__ import division
import tensorflow as tf
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import get_datasets, EarlyStopper
from model import Model

parser = ArgumentParser(
    description="Best model for Interactive Medical Image Segmentation",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--data-path", type=str, default="./", help="path to dataset"
)

# Known hyperparams
parser.add_argument(
    "--max-steps",
    type=int,
    default=50000,
    help="maximum no. of training steps",
)

# Unknown hyperparams
parser.add_argument("--random-seed", type=int, default=5, help="random seed")
parser.add_argument(
    "--val-split",
    type=float,
    default=0.2,
    help="fraction of data to use as validation (deprecated)",
)
parser.add_argument(
    "--test-split",
    type=float,
    default=0.2,
    help="fraction of data to use as test data (deprecated)",
)
parser.add_argument(
    "--early-stop-diff",
    type=float,
    default=5e-7,
    help="minimum change for early stopping",
)
parser.add_argument(
    "--early-stop-steps",
    type=int,
    default=0,
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
parser.add_argument(
    "--save-dir",
    type=str,
    default="./model.ckpt",
    help="where to store Tensorflow model",
)

args = parser.parse_args()

tf.set_random_seed(args.random_seed)

print("Loading dataset...")
data = get_datasets(args.data_path, args.val_split, args.test_split, 32)
print("Dataset loaded")

print("Building graph...")
model = Model(
    data,
    tf.train.AdamOptimizer(learning_rate=0.0008408132388618728),
    0.003683848079337278,
    0.6275728419832726,
    tf.nn.elu,
    56.89889776692406,
    45.40359136227982,
    0.10805612575300722,
)
print("Graph built")

saver = tf.train.Saver()

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

    print("Saving the model")
    save_path = saver.save(sess, args.save_dir)
    print("Model saved in path: %s" % save_path)
