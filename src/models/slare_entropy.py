import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
from models import slare_disc


@tfbp.default_export
class SLARE_Entropy(slare_disc):
    default_hparams = {
        **slare_disc.default_hparams,
    }

    def p_Y(self, x):
        """PDF estimate of Y = f(X) + E."""
        e = self.P_E.sample(tf.shape(x)[0])
        y = self.f(x) + e
        # Estimate of the ratio p_Y|X / p_Y.
        tr = self.T_ratio(tf.concat([x, y], 1))
        return self.p_E(e) / tr
