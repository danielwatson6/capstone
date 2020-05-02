import researchflow as rf
import tensorflow as tf
import tensorflow.keras.layers as tfkl

from models import mi as MI


@rf.export
class MIDisc(MI):
    """Discriminator-based MI bounder boilerplate."""

    @staticmethod
    def hparams(hp):
        MI.hparams(hp)
        hp.Int("disc_hidden_size", 128, 1024, default=128, sampling="log")
        hp.Int("disc_num_hidden", 1, 3, default=1)
        hp.Boolean("disc_iter", default=False)
        hp.Choice("disc_opt", ["sgd", "adam"], default="sgd")
        hp.Float("disc_lr", 5e-4, 5e-2, default=1e-3, sampling="log")

    def __init__(self, **kw):
        super().__init__(**kw)

        # Discriminator neural network.
        self.T = tf.keras.Sequential()
        for _ in range(self.hp.disc_num_hidden):
            self.T.add(tfkl.Dense(self.hp.disc_hidden_size, activation=tf.math.tanh))
        self.T.add(tfkl.Dense(1, activation=tf.math.softplus))

        # Optimizer.
        if self.hp.disc_opt == "adam":
            self.disc_opt = tf.optimizers.Adam(self.hp.disc_lr)
        else:
            self.disc_opt = tf.optimizers.SGD(self.hp.disc_lr)

    def I_pos(self, batch):
        """Compute the mutual information term for positive samples."""
        return tf.reduce_mean(tf.math.log(self.T(batch)))

    def I_neg(self):
        """Compute the mutual information term for negative samples."""
        raise NotImplementedError
