import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
from models import mi as MI


@tfbp.default_export
class LB_AE(MI):
    """Autoencoder."""

    default_hparams = {
        **MI.default_hparams,
        "dec_hidden": [128],
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        # Decoder neural network.
        self.dec = tf.keras.Sequential()
        for hs in self.hp.dec_hidden:
            self.dec.add(tfkl.Dense(hs, activation=tf.math.tanh))
        # TODO: abstract the output dim to the encoder's input dim.
        if self.hp.gaussian:
            raise ValueError("LB_AE does not support a Gaussian encoder.")
        else:
            self.dec.add(tfkl.Dense(28 * 28, activation=tf.math.sigmoid))

    def I(self, x):
        """Mutual information UP TO AN INTRACTABLE CONSTANT."""
        y = self.enc.p_yGx_sample(x)
        return tf.reduce_mean((self.dec(y) - x) ** 2)

    def train_step_fixed_enc(self, x):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(self.dec.trainable_weights)
            loss = -self.I(x)

        grads = g.gradient(loss, self.dec.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.dec.trainable_weights))
        return loss, -loss

    def train_step(self, x):
        with tf.GradientTape() as g:
            loss = -self.I(x)

        grads = g.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))
        return loss, -loss

    def valid_step(self, x):
        mi = self.I(x)
        return -mi, mi
