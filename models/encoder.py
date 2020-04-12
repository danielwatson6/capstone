import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl

import boilerplate as tfbp


@tfbp.default_export
class Encoder(tfbp.Model):
    """Encoder boilerplate."""

    default_hparams = {
        "enc_hidden": [128],
        "latent_size": 10,
        "var_eps": 0.1,
        "gaussian": 0.0,
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        # Forward pass neural network.
        self.f = tf.keras.Sequential()
        if self.hp.gaussian:
            # Shifting by some mean does not affect the mutual information. The
            # covariance matrix induced by this linear transformation is A A^T.
            #
            # However, the mutual information grows unbounded when the support of the
            # data distribution isn't bounded, so we can instead specify some MI (the
            # value of `self.hp.gaussian`) and set A to make this the encoder's MI.
            a_sq = self.hp.var_eps * (
                tf.math.exp(2.0 * self.hp.gaussian / self.hp.latent_size) - 1.0
            )
            self.a = a_sq ** 0.5
            self.f.add(tfkl.Lambda(lambda x: self.a * x))
        else:
            for hs in self.hp.enc_hidden:
                self.f.add(tfkl.Dense(hs, activation=tf.math.tanh))
            self.f.add(tfkl.Dense(self.hp.latent_size, activation=tf.math.tanh))

        # Constant: H[ε].
        self.H_eps = (
            np.log(2 * np.pi * np.e * self.hp.var_eps) * self.hp.latent_size / 2
        )

    def p_eps_sample(self, n=1):
        """Sample from P_ε."""
        shape = [n, self.hp.latent_size]
        return tf.random.normal(shape, stddev=self.hp.var_eps ** 0.5)

    def p_eps(self, y):
        """Evaluate p_ε(y)."""
        z = (2 * np.pi * self.hp.var_eps) ** (self.hp.latent_size / 2)
        return (
            tf.math.exp(-tf.reduce_sum(y ** 2, axis=-1) / (2 * self.hp.latent_size)) / z
        )

    def p_yGx_sample(self, x):
        """Sample from P_{Y|x}."""
        return self.f(x) + self.p_eps_sample(n=tf.shape(x)[0])

    def p_yGx(self, y, fx):
        """Evaluate p_{Y|x}(y)."""
        return self.p_eps(y - fx)
