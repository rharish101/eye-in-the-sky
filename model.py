"""Model for Interactive Medical Image Segmentation."""
import tensorflow as tf
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix
from utils import CLASSES, EarlyStopper
from dataset_rot import get_colours
from datetime import datetime
import numpy as np
import cv2
from io import BytesIO
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class Model(object):
    """Model for Interactive Medical Image Segmentation."""

    def __init__(
        self,
        data,
        optimizer,
        weight_decay,
        dropout,
        activation=tf.nn.relu,
        lmbda=0.1,
        sigma=1.0,
        t_0=0.6,
        build=True,
    ):
        """Initialize the graph with the given data.

        Args:
            data(dict): The dictionary obtained from utils.data
            optimizer(`tf.train.Optimizer`): The optimizer instance
            weight_decay(float): L2 regularisation scale
            dropout(float): dropout probability
            activation(callable): The activation function to be applied
                (None value disables this)
            build(bool): Whether to build the graph
        """
        # Sanity check
        if dropout <= 0 or dropout > 1:
            raise ValueError(
                "Invalid value for dropout (given: {})".format(dropout)
            )

        self.data = data
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.optimizer = optimizer
        self.activation = activation

        # Grid Search
        self._lmbda = lmbda
        self._sigma = sigma
        # T_1 is ignored, as we're using softmax and checking for best value
        self._t_0 = t_0

        # Colour mapping
        self._colours = get_colours(data["path"] + "gt/")

        if build:
            self._build_graph()

    def _conv_layer(
        self,
        inputs,
        filters,
        kernel_size,
        dilation_rate,
        scope,
        activation=tf.nn.relu,
        reuse=False,
    ):
        """2D convolution layer with batch normalization.

        Args:
            inputs(`tf.Tensor`): The inputs to the layer
            filters(int): The filters/channels in the output
            kernel_size(int, iterable of int): The size of the kernel (integer
                value means same size in both x & y)
            dilation_rate(int, iterable of int): The dilation rate (integer
                value means same rate in both x & y)
            scope(str): The name for the layer's scope
            activation(callable): The activation function to be applied
                (None value disables this)

        Returns:
            `tf.Tensor`: The output of the layer

        """
        with tf.variable_scope(scope, reuse=reuse):
            conv = tf.layers.conv2d(
                inputs,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                padding="same",
                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                    self.weight_decay
                ),
            )
            norm = tf.layers.batch_normalization(conv)
            if activation is not None:
                return activation(norm)
            else:
                return norm

    def _model_func(self, inputs, activation=tf.nn.relu, reuse=False):
        """Create model and return model output with and without softmax.

        Args:
            inputs(`tf.Tensor`): The inputs to the model
            activation(callable): The activation function to be applied
                (None value disables this)

        Returns:
            `tf.Tensor`: The output of the model (w. softmax)
            `tf.Tensor`: The output of the model (w/o. softmax)

        """
        # Block 1
        z = self._conv_layer(
            inputs,
            filters=64,
            kernel_size=3,
            dilation_rate=1,
            scope="b11",
            activation=activation,
            reuse=reuse,
        )
        p1 = self._conv_layer(
            z,
            filters=64,
            kernel_size=3,
            dilation_rate=1,
            scope="b12",
            activation=activation,
            reuse=reuse,
        )

        # Block 2
        z = self._conv_layer(
            p1,
            filters=64,
            kernel_size=3,
            dilation_rate=2,
            scope="b21",
            activation=activation,
            reuse=reuse,
        )
        p2 = self._conv_layer(
            z,
            filters=64,
            kernel_size=3,
            dilation_rate=2,
            scope="b22",
            activation=activation,
            reuse=reuse,
        )

        # Block 3
        z = self._conv_layer(
            p2,
            filters=64,
            kernel_size=3,
            dilation_rate=4,
            scope="b31",
            activation=activation,
            reuse=reuse,
        )
        z = self._conv_layer(
            z,
            filters=64,
            kernel_size=3,
            dilation_rate=4,
            scope="b32",
            activation=activation,
            reuse=reuse,
        )
        p3 = self._conv_layer(
            z,
            filters=64,
            kernel_size=3,
            dilation_rate=4,
            scope="b33",
            activation=activation,
            reuse=reuse,
        )

        # Block 4
        z = self._conv_layer(
            p3,
            filters=64,
            kernel_size=3,
            dilation_rate=8,
            scope="b41",
            activation=activation,
            reuse=reuse,
        )
        z = self._conv_layer(
            z,
            filters=64,
            kernel_size=3,
            dilation_rate=8,
            scope="b42",
            activation=activation,
            reuse=reuse,
        )
        p4 = self._conv_layer(
            z,
            filters=64,
            kernel_size=3,
            dilation_rate=8,
            scope="b43",
            activation=activation,
            reuse=reuse,
        )

        # Block 5
        z = self._conv_layer(
            p4,
            filters=64,
            kernel_size=3,
            dilation_rate=16,
            scope="b51",
            activation=activation,
            reuse=reuse,
        )
        z = self._conv_layer(
            z,
            filters=64,
            kernel_size=3,
            dilation_rate=16,
            scope="b52",
            activation=activation,
            reuse=reuse,
        )
        p5 = self._conv_layer(
            z,
            filters=64,
            kernel_size=3,
            dilation_rate=16,
            scope="b53",
            activation=activation,
            reuse=reuse,
        )

        # Block 6
        z = tf.concat([p1, p2, p3, p4, p5], axis=-1)
        z = tf.cond(
            self._is_train,
            lambda: tf.nn.dropout(z, keep_prob=self.dropout),
            lambda: z,
        )
        z = self._conv_layer(
            z,
            filters=128,
            kernel_size=1,
            dilation_rate=1,
            scope="b63",
            activation=activation,
            reuse=reuse,
        )
        z = tf.cond(
            self._is_train,
            lambda: tf.nn.dropout(z, keep_prob=self.dropout),
            lambda: z,
        )
        out = self._conv_layer(
            z,
            filters=CLASSES,
            kernel_size=1,
            dilation_rate=1,
            scope="b65",
            activation=None,
            reuse=reuse,
        )
        return tf.nn.softmax(out), out

    def _get_loss(self, x, y, soft_out, out):
        """Get loss function through weights, phi and psi.

        Args:
            x(`tf.Tensor`): Input to the model
            y(`tf.Tensor`): Target for the model
            soft_out(`tf.Tensor`): Model output (w. softmax)
            out(`tf.Tensor`): Model output (w/o. softmax)

        Returns:
            `tf.Tensor`: The loss for the network

        """
        max_out = tf.reduce_max(soft_out, axis=-1)
        # Best output should be at least t_0
        weights = tf.where(
            max_out < self._t_0, tf.zeros_like(max_out), tf.ones_like(max_out)
        )
        phi = tf.reduce_sum(
            weights
            * tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out),
            axis=[1, 2],
        )

        target = tf.argmax(y, axis=-1)
        shape = x.shape
        # Get tensor of 2D indices
        indices = tf.stack(
            tf.meshgrid(tf.range(shape[1]), tf.range(shape[2]), indexing="ij"),
            axis=-1,
        )

        def body(psi, i):
            """Calculate the psi energy function value for pixel i."""
            i_x = tf.to_int32(i / shape[2])
            i_y = tf.to_int32(i % shape[2])
            exp = tf.exp(
                -1.0
                * (x[:, i_x : (i_x + 1), i_y : (i_y + 1), :] - x) ** 2
                / (2 * self._sigma ** 2)
            )
            # d_ij: The Euclidean distance b/w i and j
            dist = tf.sqrt(
                tf.to_float(
                    tf.reduce_sum(((i_x, i_y) - indices) ** 2, axis=-1)
                )
            )
            # mask for: y_i != y_j
            mask = tf.where(
                target[:, i_x, i_y] != target[:, :, :],
                tf.zeros_like(target),
                tf.ones_like(target),
            )
            psi += tf.reduce_sum(
                (tf.reduce_sum(exp, axis=-1) * tf.to_float(mask)) / dist,
                axis=[1, 2],
            )
            return psi, i + 1

        psi = tf.zeros([shape[0]])
        i = tf.constant(0)
        psi, _ = tf.while_loop(
            lambda psi, i: i < shape[1] * shape[2], body, (psi, i)
        )
        return tf.reduce_mean(phi + self._lmbda * psi)

    def _build_graph(self):
        self._is_train = tf.placeholder(shape=(), dtype=tf.bool)

        def train_graph():
            x, y = self.data["iterators"]["train"].get_next()
            soft_out, out = self._model_func(x, activation=self.activation)
            return self._get_loss(x, y, soft_out, out), soft_out, y

        def test_graph():
            x, y = self.data["iterators"]["test"].get_next()
            soft_out, out = self._model_func(x, activation=self.activation, reuse=tf.AUTO_REUSE)
            return self._get_loss(x, y, soft_out, out), soft_out, y

        self._loss, soft_out, y = tf.cond(self._is_train, train_graph, test_graph)
        tf.summary.scalar("loss", self._loss)

        with tf.variable_scope("metrics") as scope:
            # Get labels and predictions as [batch_size, length], where
            # elements are labels of pixels
            pred = tf.layers.flatten(tf.argmax(soft_out, axis=-1))
            labels = tf.layers.flatten(tf.argmax(y, -1))
            # True indicates that the pixel does not have white colour
            # This mask is obtained on labels only
            mask = tf.not_equal(labels, self._colours.index((255, 255, 255)))
            # Choose those pixels that DO NOT have white colour
            # NOTE: This destroys the batch axis
            labels = tf.boolean_mask(labels, mask)
            # The same mask is used for predictions, to ignore the exact
            # same pixels
            pred = tf.boolean_mask(pred, mask)

            _, self._acc_op = tf.metrics.accuracy(
                labels=labels, predictions=pred
            )
            tf.summary.scalar("acc", self._acc_op)

            _, self._kappa_op = tf.contrib.metrics.cohen_kappa(
                labels=labels, predictions_idx=pred, num_classes=CLASSES
            )
            tf.summary.scalar("kappa", self._kappa_op)

            _, self._conf_mat = _streaming_confusion_matrix(
                labels=labels, predictions=pred, num_classes=CLASSES
            )
            tf.summary.image("confusion", self._get_mat_img(self._conf_mat))

            metrics = tf.contrib.framework.get_variables(
                scope, collection=tf.GraphKeys.LOCAL_VARIABLES
            )
            self._init_metrics = tf.variables_initializer(metrics)

        self._train_step = self.optimizer.minimize(self._loss)
        self._merged_summ = tf.summary.merge_all()

    def _get_mat_img(self, conf_mat, numpy=False):
        if numpy:
            total = np.sum(conf_mat)
            rescaled = (255 * (conf_mat / np.clip(total, 1, 1e7))).astype(
                np.uint8
            )
            image = np.zeros((CLASSES * 10, CLASSES * 10))
            for i in range(CLASSES * 10):
                for j in range(CLASSES * 10):
                    image[i, j] = rescaled[i // 10, j // 10]
            return image
        else:
            total = tf.reduce_sum(conf_mat, axis=0)
            rescaled = tf.cast(
                255 * (conf_mat / tf.clip_by_value(total, 1, 1e7)), tf.uint8
            )
            image = tf.image.resize_images(
                tf.reshape(rescaled, (1, CLASSES, CLASSES, 1)),
                [CLASSES * 10, CLASSES * 10],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            )
            return image

    def evaluate(self, mode, sess=None):
        """Evaluate the model on the validation/test dataset.

        Mode should be one of "val" (validation) or "test" (testing)
        """
        mode = mode.lower().strip()
        if mode not in ["val", "test"]:
            raise ValueError("Mode should be one of 'val' or 'test'")
        if not hasattr(self, "_sess"):
            if sess is None:
                raise ValueError(
                    "Session not set. Please pass a session argument"
                )
            else:
                self._sess = sess

        self._sess.run(self.data["init_ops"][mode])
        self._sess.run(self._init_metrics)
        count = 0
        curr_loss = 0
        try:
            while True:
                loss_diff, curr_acc, curr_kappa, curr_conf = self._sess.run(
                    [self._loss, self._acc_op, self._kappa_op, self._conf_mat],
                    feed_dict={self._is_train: False},
                )
                curr_loss += loss_diff
                count += 1
        except tf.errors.OutOfRangeError:
            # End of dataset
            curr_loss /= count
            conf_mat_img = self._get_mat_img(curr_conf, numpy=True)

        if mode == "val":
            # Add Tensorboard summary for validation loss and metrics manually
            summary = tf.Summary()
            summary.value.add(tag="loss", simple_value=curr_loss)
            self._val_writer.add_summary(summary, self._step)

            summary = tf.Summary()
            summary.value.add(tag="metrics/acc", simple_value=curr_acc)
            self._val_writer.add_summary(summary, self._step)

            summary = tf.Summary()
            summary.value.add(tag="metrics/kappa", simple_value=curr_kappa)
            self._val_writer.add_summary(summary, self._step)

            str_io = BytesIO()
            plt.imsave(str_io, conf_mat_img, format="png", cmap="gray")
            img_sum = tf.Summary.Image(
                encoded_image_string=str_io.getvalue(),
                height=conf_mat_img.shape[0],
                width=conf_mat_img.shape[1],
            )
            summary = tf.Summary()
            summary.value.add(tag="metrics/confusion", image=img_sum)
            self._val_writer.add_summary(summary, self._step)

            print(
                "Step: {:5d}, Val. Loss: {:12.4f}, Val. Acc: {:.4f}, "
                "Val. Kappa: {:.4f}".format(
                    self._step + 1, curr_loss, curr_acc, curr_kappa
                )
            )

            if self.stopper is not None:
                if self.stop_on == "loss":
                    self.stopper.update(curr_loss)
                elif self.stop_on == "acc":
                    self.stopper.update(curr_acc)
        else:
            print(
                "Test Loss: {:12.4f}, Test Acc.: {:.4f}, "
                "Test Kappa: {:.4f}".format(curr_loss, curr_acc, curr_kappa)
            )
            np.save("./conf_mat.npy", conf_mat_img)
            print("Test Confustion Matrix saved as ./conf_mat.npy")

        return curr_loss, curr_acc, curr_kappa, curr_conf

    def inference(self, path, sess_options=tf.ConfigProto()):
        """Build inference graph and runs on the actual test dataset.

        When inference is required, this model should be initialized without
        the graph being built. Note than this should not be called inside a
        `tf.Session()`, as it creates one by itself.

        Args:
            path: path to the saved model
            sess_options: options for the `tf.Session` invoked here
        """
        self._is_train = tf.constant(False)
        x, name = self.data["iterators"]["actual"].get_next()
        soft_out, _ = self._model_func(x, activation=self.activation)

        _, indices = tf.nn.top_k(soft_out, k=2)
        first, second = tf.unstack(indices, axis=-1)
        pred = tf.where(
            tf.not_equal(first, self._colours.index((255, 255, 255))),
            first,
            second,
        )

        loader = tf.train.Saver()

        images = []
        names = []
        with tf.Session(config=sess_options) as sess:
            loader.restore(sess, path)
            try:
                while True:
                    curr_images, curr_names = sess.run([pred, name])
                    images += list(curr_images)
                    names += list(curr_names)
            except tf.errors.OutOfRangeError:
                # End of dataset
                pass

        for img, name in zip(images, names):
            new_img = np.zeros((*(img.shape[:2]), 3))
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    new_img[i, j] = self._colours[img[i, j]]
            cv2.imwrite(
                name.decode("utf8") + "_result.png", new_img[:, :, ::-1]
            )

    def train(
        self,
        sess,
        max_steps,
        logdir,
        log_steps=0,
        stopper=None,
        stop_on="loss",
    ):
        """Train the model on the training dataset.

        Args:
            sess(`tf.Session`/`tf.InteractiveSession`): The TensorFlow session
            max_steps(int): Maximum number of steps to train for
            log_steps(int): Steps after which to test on validation
                (0 value disables this)
            stopper(`utils.EarlyStopper`): A stopper for early stopping on
                validation data (None value disables this)
            stop_on(str): What to use early stopping for
                (available: "loss", "acc")
        """
        if max_steps < 0:
            raise ValueError("Max. steps must be non-negative")
        if log_steps < 0:
            raise ValueError("Logging steps must be non-negative")

        self._sess = sess

        sess.run(tf.global_variables_initializer())
        if logdir[-1] != "/":
            logdir += "/"
        logdir = logdir + datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
        self.train_writer = tf.summary.FileWriter(logdir + "/train")
        self._val_writer = tf.summary.FileWriter(logdir + "/val")

        if stopper is None:
            self.stopper = None
        elif type(stopper) != EarlyStopper:
            raise ValueError(
                "Given stopper is not of type `utils.EarlyStopper`"
            )
        else:
            self.stopper = stopper
            stop_on = stop_on.lower().strip()
            if stop_on not in ["loss", "acc"]:
                raise ValueError(
                    "Early stopping should be on one of 'loss' or 'acc'"
                )
            self.stop_on = stop_on

        for i in range(max_steps):
            self._step = i
            sess.run(self._init_metrics)
            summary, _ = sess.run(
                [self._merged_summ, self._train_step],
                feed_dict={self._is_train: True},
            )
            self.train_writer.add_summary(summary, i)

            # Validation data testing
            if log_steps != 0 and i % log_steps == 0:
                self.evaluate("val")
                if self.stopper is not None and self.stopper.stop:
                    print("Stopping")
                    break

            # Test data
            if log_steps != 0 and i % (4 * log_steps) == 0:
                self.evaluate("test")
