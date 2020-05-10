import researchflow as rf
import tensorflow as tf

from models import ub_disc as UBDisc


@rf.export
class UB_MINE(UBDisc):
    """Nguyen-Wainwright-Jordan MI upper bounder."""

    @staticmethod
    def hparams(hp):
        UBDisc.hparams(hp)
        hp.Float("decay_rate", 0.1, 0.9, default=0.5)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.ema = tf.Variable(0.0, trainable=False)

    def D_neg(self, y):
        if self.method != "train":
            return -tf.math.log(tf.reduce_mean(self.T(y)))

        # Naive biased gradients look like this:
        #   d/dt -log (E_x T_t(x))
        # = -d/dt (E_x T_t(x)) / E_x T_t(x)
        #
        # We want the gradient computation to yield this:
        # -d/dt (E_x T_t(x)) / EMA(E_x T_t(x))
        e_t = tf.reduce_mean(self.T(y))
        if self.ema == 0:
            self.ema.assign(e_t)
        else:
            self.ema.assign(
                self.hp.decay_rate * e_t + (1.0 - self.hp.decay_rate) * self.ema
            )
        return -e_t / tf.stop_gradient(self.ema)
