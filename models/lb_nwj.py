import numpy as np
import researchflow as rf
import tensorflow as tf

from models import lb_disc as LBDisc


@rf.export
class LB_NWJ(LBDisc):
    """Nguyen-Wainwright-Jordan MI lower bounder."""

    def I_neg(self, xy):
        return -tf.reduce_mean(self.T(xy)) / np.e
