import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
from models import slare
from utils import mish, tanh_squeezed


@tfbp.default_export
class SLARE_Disc(slare):
    """Abstract SLARE model with a discriminator."""

    default_hparams = {
        **slare.default_hparams,
        "disc": "dv",  # dv, gan, ganhack, fdiv
        "disc_hidden_sizes": [256],
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        self.T = tf.keras.Sequential()
        for size in self.hparams.hidden_sizes:
            self.T.add(tfkl.Dense(size, activation=mish))

        activation = tf.math.softplus
        if self.hparams.disc == "gan":
            activation = tanh_squeezed
        self.T.add(tfkl.Dense(1, activation=activation))

    def T_ratio(self, inputs):
        """PDF ratio estimate as a transformation of the discriminator."""
        t = self.T(inputs)
        if self.hparams.disc == "gan":
            return t / (1.0 - t)
        return t

    def loss_pos(self, inputs):
        """Discriminator loss for positive examples."""
        return tf.reduce_mean(tf.math.log(self.T(inputs)))

    def loss_neg(self, inputs):
        """Discriminator loss for negative examples."""
        if self.hparams.disc == "gan":
            return tf.reduce_mean(tf.math.log(1.0 - self.T(inputs)))
        if self.hparams.disc == "ganhack":
            return -tf.reduce_mean(tf.math.log(self.T(inputs)))
        if self.hparams.disc == "fdiv":
            return -tf.reduce_mean(tf.math.exp(-1) * self.T(inputs))
        return -tf.math.log(tf.reduce_mean(self.T(inputs)))

    def loss_disc(self, inputs_pos, inputs_neg):
        """Total loss of the discriminator."""
        return self.loss_pos(inputs_pos) + self.loss_neg(inputs_neg)

    @tfbp.runnable
    def train(self, data_loader):
        train_data, test_data = data_loader.load()

        for _ in range(self.hparams.epochs):
            ...
