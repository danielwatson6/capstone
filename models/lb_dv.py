import researchflow as rf
import tensorflow as tf

from models import lb_disc as LBDisc


@rf.export
class LB_DV(LBDisc):
    """Donsker-Varadhan MI lower bounder."""

    def I_neg(self, xy):
        return -tf.math.log(tf.reduce_mean(self.T(xy)))
