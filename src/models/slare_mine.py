import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
from models import slare_disc


@tfbp.default_export
class SLARE_MINE(slare_disc):
    default_hparams = {
        **slare_disc.default_hparams,
        "slare_mi": False,
    }

    def p_Y(self, x):
        """PDF estimate of Y = f(X) + E."""
        e = self.P_E.sample(tf.shape(x)[0])
        y = self.f(x) + e
        # Estimate of the ratio p_Y|X / p_Y.
        tr = self.T_ratio(tf.concat([x, y], 1))
        return self.p_E(e) / tr

    def I_XY(self, x):
        # MINE yields two ways to estimate the mutual information:
        # 1. Use the fact that the total discriminator loss is a lower bound to MI.
        # 2. Use the PDF ratio estimate yielded by the discriminator with SLARE.
        if self.hparams.slare_mi:
            x_pos, x_neg = tf.split(x, 2)
            y_pos = self.f(x_pos)
            xy_pos = tf.concat([x_pos, y_pos], 1)
            xy_neg = tf.concat([x_neg, y_pos], 1)
            return self.loss_disc(xy_pos, xy_neg)
        return super().I_XY(x)
