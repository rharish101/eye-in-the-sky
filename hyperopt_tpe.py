#!/usr/bin/env python3
"""HyperOpt Tree of Parzen Estimators method."""
from __future__ import print_function
from __future__ import division
import pickle
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
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
parser.add_argument(
    "--max-steps", type=int, default=1000, help="maximum no. of training steps"
)

# Unknown hyperparams
parser.add_argument("--random-seed", type=int, default=5, help="random seed")
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

parser.add_argument(
    "--max-iters",
    type=int,
    default=100,
    help="max steps for hyperparameter search using hyperopt",
)
parser.add_argument(
    "--trial-save",
    type=str,
    default="./",
    help="path to save the pickle of trials",
)

args = parser.parse_args()

tf.set_random_seed(args.random_seed)

space = {
    # Learning Rate
    "lr": hp.uniform("lr_rate_mult", 0.00001, 0.001),
    # L2 weight decay:
    "weight_decay": hp.uniform("weight_decay", 5e-5, 5e-3),
    # Batch size fed for each gradient update
    "batch_size": hp.choice("batch_size", [16, 32, 64]),
    # Choice of optimizer:
    "optimizer": hp.choice("optimizer", ["Adam", "Adagrad", "RMSprop"]),
    # Uniform distribution in finding appropriate dropout values, conv layers
    "dropout_drop_proba": hp.uniform("dropout_proba", 0.0, 0.7),
    # Activations that are used everywhere
    "activation": hp.choice("activation", ["relu", "elu"]),
    # Other scaling hyperparameters
    "t_0": hp.uniform("t_0", 0.0, 1.0),
    "sigma": hp.uniform("sigma", 0.0, 100.0),
    "lmbda": hp.uniform("lmbda", 0.0, 100.0),
}


def interpret(space):
    lr = space["lr"]
    batch_size = space["batch_size"]
    optimizer = space["optimizer"]
    drop_rate = space["dropout_drop_proba"]
    activation = space["activation"]
    weight_decay = space["weight_decay"]

    if optimizer == "Adam":
        opt = tf.train.AdamOptimizer(learning_rate=lr)
    elif optimizer == "Adagrad":
        opt = tf.train.AdagradOptimizer(learning_rate=lr)
    else:
        opt = tf.train.RMSPropOptimizer(learning_rate=lr)

    if activation == "relu":
        act = tf.nn.relu
    else:
        act = tf.nn.elu

    # Load Dataset
    data = get_datasets(
        args.data_path, args.val_split, args.test_split, batch_size
    )

    # Building the Graph
    model = Model(
        data,
        opt,
        weight_decay,
        drop_rate,
        act,
        space["lmbda"],
        space["sigma"],
        space["t_0"],
    )

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
            result = {
                # Loss is the negative accuracy, so that we get the maximum accuracy
                "status": STATUS_FAIL,
                "space": space,
            }
            return result

        loss, acc, kappa, conf_mat = model.evaluate("val")

    result = {
        # Loss is the negative accuracy, so that we get the maximum accuracy
        "loss": -acc,
        "validation loss": loss,
        "validation accuracy": acc,
        "status": STATUS_OK,
        "space": space,
    }
    tf.reset_default_graph()

    return result


trials = Trials()

best = fmin(
    fn=interpret,
    space=space,
    algo=tpe.suggest,
    trials=trials,
    max_evals=args.max_iters,
)

# The trials database now contains 100 entries, it can be saved/reloaded with
# pickle or another method
pickle.dump(trials, open("{}saved_trials.p".format(args.trial_save), "wb"))
# trials = pickle.load(open("saved_trials.p", "rb"))

print("Found minimum:")
print(best)
print("")
