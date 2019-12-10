import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow_probability import distributions as tfd

import boilerplate as tfbp
from utils import mish, tanh_restricted


@tfbp.default_export
class SLARE(tfbp.Model):
    """Abstract SLARE model."""

    default_hparams = {
        "output_size": 2,
        "target_mi": 1.0,
        "hidden_sizes": [256],
        "square_loss": False,
        "beta": 1e-3,
        "epochs": 10,
        "num_steps_per_eval": 100,
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        if square_loss:
            beta = self.hparams.beta
        else:
            # We want H[P_epsilon] = H[Uniform] - target_mi = - target_mi
            beta = 0.5 * tf.math.exp(-target_mi / self.hparams.output_size)

        self.P_E = tfd.Uniform(
            low=[-beta] * self.hparams.output_size,
            high=[beta] * self.hparams.output_size,
        )

        self.f = tf.keras.Sequential()
        for size in self.hparams.hidden_sizes:
            self.f.add(tfkl.Dense(size, activation=mish))
        self.f.add(
            tfkl.Dense(self.hparams.output_size, activation=tanh_restricted(beta))
        )

    def p_E(self, y):
        """PDF of E."""
        return self.P_E.prob(y)

    def H_E(self):
        """Entropy of E."""
        return self.P_E.entropy()

    def p_Y(self, x):
        """PDF estimate of Y = f(X) + E."""
        raise NotImplementedError

    def H_Y(self, x):
        """Entropy estimate of Y."""
        return -tf.reduce_mean(tf.math.log(self.p_Y(x)))

    def I_XY(self, x):
        """Mutual information estimate between X and Y."""
        return self.H_Y(x) - self.H_E()

    def loss(self, x):
        if square_loss:
            return (self.I_XY(x) - self.target_mi) ** 2
        return -self.I_XY(x)
