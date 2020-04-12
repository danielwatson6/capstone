import numpy as np
import tensorflow as tf

import boilerplate as tfbp
from models import lb_disc as LBDisc


@tfbp.default_export
class LB_DV(LBDisc):
    """Nguyen-Wainwright-Jordan MI lower bounder."""

    def I_neg(self, xy):
        return -tf.reduce_mean(self.T(xy)) / np.e
