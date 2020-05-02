import researchflow as rf
import tensorflow as tf

from models import ub_disc as UBDisc


@rf.export
class UB_NWJ(UBDisc):
    """Nguyen-Wainwright-Jordan MI upper bounder."""

    def I_neg(self, y):
        return -tf.math.log(tf.reduce_mean(self.T(y)))
