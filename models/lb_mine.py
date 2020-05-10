import researchflow as rf
import tensorflow as tf

from models import lb_disc as LBDisc


@rf.export
class LB_MINE(LBDisc):
    """Donsker-Varadhan MI lower bounder."""

    @staticmethod
    def hparams(hp):
        LBDisc.hparams(hp)
        hp.Float("decay_rate", 0.1, 0.9, default=0.5)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.ema = tf.Variable(0.0, trainable=False)

    def I_neg(self, xy):
        if self.method != "train":
            return -tf.math.log(tf.reduce_mean(self.T(xy)))

        # Naive biased gradients look like this:
        #   d/dt -log (E_x T_t(x))
        # = -d/dt (E_x T_t(x)) / E_x T_t(x)
        #
        # We want the gradient computation to yield this:
        # -d/dt (E_x T_t(x)) / EMA(E_x T_t(x))
        e_t = tf.reduce_mean(self.T(xy))
        if self.ema == 0:
            self.ema.assign(e_t)
        else:
            self.ema.assign(self.hp.decay_rate * e_t + (1.0 - self.hp.decay_rate) * ema)
        return -e_t / tf.stop_gradient(ema)
