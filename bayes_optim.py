#!/usr/bin/env python3
"""Bayesian Optimization"""
from __future__ import print_function
from __future__ import division
import pickle
from bayes_opt import BayesianOptimization
import tensorflow as tf
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import traceback
from utils import get_datasets, EarlyStopper
from model import Model

parser = ArgumentParser(
    description="Bayesian Optimization for Medical Image Segmentation",
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
    "--expl-iters",
    type=int,
    default=20,
    help="max steps for exploration of hyperparameters using bayesian opt.",
)
parser.add_argument(
    "--opt-iters",
    type=int,
    default=100,
    help="max steps for hyperparameter search using bayesian opt.",
)
parser.add_argument(
    "--opt-save",
    type=str,
    default="./",
    help="path to save the pickle of optimal hyperparams",
)

args = parser.parse_args()

tf.set_random_seed(args.random_seed)

space = {
    # Learning Rate
    'lr': (0.00001, 0.001),
    # L2 weight decay:
    'weight_decay': (5e-5, 5e-3),
    # Uniform distribution in finding appropriate dropout values, conv layers
    'dropout_drop_proba': (0.01, 0.7),
    # Other scaling hyperparameters
    't_0': (0.0, 1.0),
    'sigma': (0.001, 100.0),
    'lmbda': (0.0, 100.0)
}

def interpret(**space):
    lr = space['lr']
    batch_size = 32
    drop_rate = space['dropout_drop_proba']
    weight_decay = space['weight_decay']

    opt = tf.train.AdamOptimizer(learning_rate=lr)
    act = tf.nn.elu

    # Load Dataset
    data = get_datasets(
        args.data_path, args.val_split, args.test_split, batch_size
    )

    try:
        # Building the Graph
        model = Model(data, opt, weight_decay, drop_rate, act, space['lmbda'], space['sigma'], space['t_0'])

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
                    "space": space
                }
                return result

            loss, acc = model.evaluate("val")
    except Exception:
        print(traceback.format_exc())
        acc = 0.0

    tf.reset_default_graph()

    # We need to maximize the accuracy
    return acc

bay_opt = BayesianOptimization(
    f=interpret,
    pbounds=space,
    random_state=args.random_seed
)
bay_opt.maximize(init_points=args.expl_iters, n_iter=args.opt_iters)

# The trials database now contains 100 entries, it can be saved/reloaded with pickle or another method
pickle.dump(bay_opt.max, open("{}bayes_max.p".format(args.trial_save), "wb"))
# trials = pickle.load(open("saved_trials.p", "rb"))

print("Found minimum:")
print(bay_opt.max)
print("")
