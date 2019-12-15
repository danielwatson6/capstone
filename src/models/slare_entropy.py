import tensorflow as tf
from tensorflow_probability import distributions as tfd

import boilerplate as tfbp
from models import slare_disc


@tfbp.default_export
class SLARE_Entropy(slare_disc):
    """SLARE model with a discriminator on output space for entropy estimation."""

    default_hparams = {
        **slare_disc.default_hparams,
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.P_U = tfd.Uniform(
            low=[0.0] * self.hparams.output_size, high=[1.0] * self.hparams.output_size
        )

    def T_inputs(self, x, y):
        """Build inputs for the discriminator."""
        return y, self.P_U.sample(tf.shape(y)[0])

    def loss_inputs(self, x, y):
        """Build inputs for the main loss."""
        return y

    def p_U(self, y):
        """PDF of U."""
        return self.P_U.prob(y)

    def H_U(self):
        return tf.reduce_sum(self.P_U.entropy())

    def H_Y(self, y):
        """Entropy estimate of Y."""
        # We can take advantage of the fact that we know the exact entropy of U.
        tr = self.T_ratio(y)  # estimate of the ratio p_Y / p_U
        return -tf.reduce_mean(tf.math.log(tr)) + self.H_U()
