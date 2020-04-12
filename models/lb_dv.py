import tensorflow as tf

import boilerplate as tfbp
from models import lb_disc as LBDisc


@tfbp.default_export
class LB_DV(LBDisc):
    """Donsker-Varadhan MI lower bounder."""

    def I_neg(self, xy):
        return -tf.math.log(tf.reduce_mean(self.T(xy)))
