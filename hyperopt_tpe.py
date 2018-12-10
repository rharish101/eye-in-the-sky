# HyperOpt Tree of Parzen Estimators method
import pickle
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL

space = {
    # Learning Rate
    'lr': hp.loguniform('lr_rate_mult', 0.00001, 0.001),
    # L2 weight decay:
    'weight_decay': hp.loguniform('weight_decay', 5e-5, 5e-3),
    # Batch size fed for each gradient update
    'batch_size': hp.quniform('batch_size', 32, 256, 32),
    # Choice of optimizer:
    'optimizer': hp.choice('optimizer', ['Adam', 'Adagrad', 'RMSprop']),
    # Uniform distribution in finding appropriate dropout values, conv layers
    'dropout_drop_proba': hp.uniform('dropout_proba', 0.0, 0.7),
    # Activations that are used everywhere
    'activation': hp.choice('activation', ['relu', 'elu']),
    # Other scaling hyperparameters
    't_0': hp.uniform('t_0', 0.0, 1.0),
    'sigma': hp.uniform('sigma', 0.0, 100.0),
    'lmbda': hp.uniform('lmbda', 0.0, 100.0)
}

def interpret(space):
    lr = space['lr']
    batch_size = space['batch_size']
    optimizer = space['optimizer']
    drop_rate = space['dropout_drop_proba']
    activation = space['activation']
    weight_decay = space['weight_decay']

    if optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(learning_rate=lr)
    elif optimizer == 'Adagrad':
        opt = tf.train.AdagradOptimizer(learning_rate=lr)
    else:
        opt = tf.train.RMSPropOptimizer(learning_rate=lr)

    if activation == 'relu':
        act = tf.nn.relu
    else:
        act = tf.nn.elu

    # Load Dataset
    data = get_datasets(
        args.data_path, args.val_split, args.test_split, batch_size
    )

    # Building the Graph
    model = Model(data, optimizer, weight_decay, drop_rate, act, space['lmbda'], space['sigma'], space['t_0'])

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

    result = {
        # Loss is the negative accuracy, so that we get the maximum accuracy
        "loss": -acc,
        "validation loss": loss,
        "validation accuracy": acc,
        "status": STATUS_OK,
        "space": space
    }
    return result

def run(max_itersn trial_save):
    trials = Trials()

    best = fmin(
        fn=interpret,
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_iters
    )

    # The trials database now contains 100 entries, it can be saved/reloaded with pickle or another method
    pickle.dump(trials, open("{}saved_trials.p".format(trial_save), "wb"))
    # trials = pickle.load(open("saved_trials.p", "rb"))

    print("Found minimum:")
    print(best)
    print("")
