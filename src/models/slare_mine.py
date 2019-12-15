import tensorflow as tf

import boilerplate as tfbp
from models import slare_disc


@tfbp.default_export
class SLARE_MINE(slare_disc):
    """SLARE model following Mutual Information Neural Estimation.

    http://proceedings.mlr.press/v80/belghazi18a.html
    """

    default_hparams = {
        **slare_disc.default_hparams,
        "slare_mi": False,
    }

    def T_inputs(self, x, y):
        """Build inputs for the discriminator."""
        x_pos, x_neg = tf.split(x, 2)
        y_pos, _ = tf.split(y, 2)
        return tf.concat([x_pos, y_pos], 1), tf.concat([x_neg, y_pos], 1)

    def loss_inputs(self, x, y):
        """Build inputs for the main loss."""
        # MINE yields two ways to estimate the mutual information:
        # 1. Use the fact that the total discriminator loss is a lower bound to MI.
        # 2. Use the PDF ratio estimate yielded by the discriminator with SLARE.
        if self.hparams.slare_mi:
            return tf.concat([x, y], 1)
        return self.T_inputs(x, y)

    def H_Y(self, xy):
        """Entropy estimate of Y."""
        # We can take advantage of the fact that we know the exact entropy of E.
        tr = self.T_ratio(xy)  # estimate of the ratio p_Y|X / p_Y
        return tf.reduce_mean(tf.math.log(tr)) + self.H_E()

    def I_XY(self, xy):
        """Mutual information estimate between X and Y."""
        if self.hparams.slare_mi:
            return super().I_XY(xy)
        return -self.loss_disc(*xy)
