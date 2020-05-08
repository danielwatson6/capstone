import numpy as np
import researchflow as rf
import tensorflow as tf

from models import ub_disc as UBDisc


@rf.export
class UB_NWJ(UBDisc):
    """Nguyen-Wainwright-Jordan MI upper bounder."""

    def D_neg(self, y):
        return -tf.reduce_mean(self.T(y)) / np.e
