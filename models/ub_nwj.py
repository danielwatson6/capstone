import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
from models import ub_disc as UBDisc


@tfbp.default_export
class UB_GAN(UBDisc):
    """Nguyen-Wainwright-Jordan MI upper bounder."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.T.add(tfkl.Dense(1, activation=tf.math.softplus))

    def I_neg(self, xi):
        return -tf.reduce_mean(self.T(xi)) / np.e

    def q_y(self, y):
        return self.p_xi(y) * self.T(y) / np.e
