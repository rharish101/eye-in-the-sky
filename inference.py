#!/usr/bin/env python3
# coding: utf-8
"""Do inference to get segmentation using existing model."""
from __future__ import print_function
from __future__ import division
import tensorflow as tf
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import get_datasets
from model import Model

parser = ArgumentParser(
    description="Inference using Interactive Medical Image Segmentation",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--data-path", type=str, default="./", help="path to dataset"
)

# Unknown hyperparams
parser.add_argument("--random-seed", type=int, default=5, help="random seed")

# Miscellaneous
parser.add_argument(
    "--save-dir",
    type=str,
    default="./model.ckpt",
    help="where the Tensorflow model is stored",
)
parser.add_argument(
    "--include-white",
    action="store_true",
    help="whether to include white pixels in the output",
)

args = parser.parse_args()

tf.set_random_seed(args.random_seed)

print("Loading dataset...")
data = get_datasets(args.data_path, 0, 0, 32)
print("Dataset loaded")

model = Model(
    data,
    tf.train.AdamOptimizer(learning_rate=0.0008408132388618728),
    0.003683848079337278,
    0.6275728419832726,
    tf.nn.elu,
    1000,
    100,
    0.10805612575300722,
    build=False,
)

# Limit GPU usage
gpu_options = tf.GPUOptions(allow_growth=True)
print("Starting inference...")
model.inference(
    args.save_dir,
    tf.ConfigProto(gpu_options=gpu_options),
    include_white=args.include_white,
)
print("Inference done")
