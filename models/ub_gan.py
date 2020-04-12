import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
from models import ub_disc as UBDisc


@tfbp.default_export
class UB_GAN(UBDisc):
    """Generative adversarial network MI upper bounder."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.T.add(tfkl.Dense(1, activation=tf.math.sigmoid))

    def I_neg(self, xi):
        return tf.reduce_mean(tf.math.log(1.0 - self.T(xi)))

    def q_y(self, y):
        ty = self.T(y)
        return self.p_xi(y) * ty * tf.math.reciprocal(1.0 - ty)
