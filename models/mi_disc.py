import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
from models import mi as MI


@tfbp.default_export
class MIDisc(MI):
    """Discriminator-based MI bounder boilerplate."""

    default_hparams = {
        **MI.default_hparams,
        "disc_hidden": [128],
        "disc_iter": False,
        "disc_opt": "sgd",
        "disc_lr": 0.1,
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        # Discriminator neural network.
        self.T = tf.keras.Sequential()
        for hs in self.hp.disc_hidden:
            self.T.add(tfkl.Dense(hs, activation=tf.math.tanh))

        # Optimizer.
        if self.hp.disc_opt.lower() == "adam":
            self.disc_opt = tf.optimizers.Adam(self.hp.disc_lr)
        else:
            self.disc_opt = tf.optimizers.SGD(self.hp.disc_lr)

    def I_pos(self, batch):
        """Compute the mutual information term for positive samples."""
        return tf.reduce_mean(tf.math.log(self.T(batch)))

    def I_neg(self):
        """Compute the mutual information term for negative samples."""
        raise NotImplementedError
