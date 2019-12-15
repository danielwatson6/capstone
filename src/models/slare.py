import os

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow_probability import distributions as tfd

import boilerplate as tfbp
from utils import mish, tanh_restricted


@tfbp.default_export
class SLARE(tfbp.Model):
    """Abstract SLARE model."""

    default_hparams = {
        "output_size": 10,
        "target_mi": 10.0,
        "hidden_sizes": [256],
        "square_loss": False,
        "beta": 1e-3,
        "optimizer": "sgd",  # sgd or adam
        "lr": 0.1,
        "epochs": 3,
        "num_steps_per_eval": 100,
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        self.train_writer = tf.summary.create_file_writer(
            os.path.join(self.save_dir, "train")
        )
        self.valid_writer = tf.summary.create_file_writer(
            os.path.join(self.save_dir, "valid")
        )

        if self.hparams.square_loss:
            beta = self.hparams.beta
        else:
            # We want H[P_E] = H[Uniform] - target_mi = - target_mi
            beta = 0.5 * tf.math.exp(-self.hparams.target_mi / self.hparams.output_size)

        self.P_E = tfd.Uniform(
            low=[-beta] * self.hparams.output_size,
            high=[beta] * self.hparams.output_size,
        )

        self.f = tf.keras.Sequential()
        for size in self.hparams.hidden_sizes:
            self.f.add(tfkl.Dense(size, activation=tf.nn.relu))
        self.f.add(
            tfkl.Dense(self.hparams.output_size, activation=tanh_restricted(beta))
        )

    def loss_inputs(self, x, y):
        """Build inputs for the main loss."""
        raise NotImplementedError

    def p_E(self, y):
        """PDF of E."""
        return self.P_E.prob(y)

    def H_E(self):
        """Entropy of E."""
        return tf.reduce_sum(self.P_E.entropy())

    def p_Y(self, inputs):
        """PDF estimate of Y = f(X) + E."""
        raise NotImplementedError

    def H_Y(self, inputs):
        """Entropy estimate of Y."""
        return -tf.reduce_mean(tf.math.log(self.p_Y(inputs)))

    def I_XY(self, inputs):
        """Mutual information estimate between X and Y."""
        return self.H_Y(inputs) - self.H_E()

    def loss(self, inputs):
        """Get the mutual information estimate and the loss."""
        mi = self.I_XY(inputs)
        if self.hparams.square_loss:
            return mi, (mi - self.hparams.target_mi) ** 2
        return mi, -mi

    def print_stats(self, step, duration, estimated_mi, valid=False):
        """Print error metrics and other useful information."""
        tmi = self.hparams.target_mi
        emi = estimated_mi
        if valid:
            print("  Valid stats")
        else:
            print("Step {} ({:.4f}s)".format(step, duration))
            print("  Train stats")
        print("    target_mi={:.4f}".format(tmi))
        print("    estimated_mi={:.4f}".format(emi.numpy()))
        print("    abs_error={:.4f}".format(tf.math.abs(estimated_mi - tmi).numpy()))
        print("    cross_entropy={:.4f}".format((estimated_mi + self.H_E()).numpy()))

    @tfbp.runnable
    def eval(self, data_loader):
        valid_data = data_loader.load()
        emis = []
        for x, _ in valid_data:
            y = self.f(x) + self.P_E.sample(tf.shape(x)[0])
            emi, _ = self.loss(self.loss_inputs(x, y))
            emis.append(emi)
        tmi = self.hparams.target_mi
        mmi = tf.reduce_mean(emis)
        print("Target MI: {:.4f}".format(tmi))
        print("Mean estimated MI: {:.4f}".format(mmi.numpy()))
        print("Abs error: {:.4f}".format(tf.math.abs(mmi - tmi).numpy()))
        print("Cross entropy: {:.4f}".format((mmi + self.H_E()).numpy()))

    @tfbp.runnable
    def eval_test(self, data_loader):
        self.eval(data_loader)
