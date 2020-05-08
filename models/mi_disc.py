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
        hp.Fixed("disc_hidden_size", 1024)
        hp.Fixed("disc_num_hidden", 2)
        hp.Choice("disc_activation", ["tanh", "relu"], default="tanh")
        hp.Boolean("disc_iter", default=False)
        hp.Choice("disc_opt", ["sgd", "adam"], default="sgd")
        hp.Float("disc_lr", 5e-4, 0.1, default=1e-3, sampling="log")

    def __init__(self, **kw):
        super().__init__(**kw)

        self.g = tf.Variable(1.0, trainable=False)

        # Discriminator neural network.
        self.T = tf.keras.Sequential()
        for _ in range(self.hp.disc_num_hidden):
            self.T.add(
                tfkl.Dense(self.hp.disc_hidden_size, activation=self.hp.disc_activation)
            )
        self.T.add(tfkl.Dense(1, activation=tf.math.softplus))

        # Optimizer.
        if self.hp.disc_opt == "adam":
            self.disc_opt = tf.optimizers.Adam(self.hp.disc_lr)
        else:
            self.disc_opt = tf.optimizers.SGD(self.hp.disc_lr)
