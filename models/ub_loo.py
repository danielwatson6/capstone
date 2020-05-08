import researchflow as rf
import tensorflow as tf

from models import mi as MI


@rf.export
class UB_LOO(MI):
    """Leave-one-out MI upper bounder."""

    @staticmethod
    def hparams(hp):
        MI.hparams(hp)
        hp.Fixed("div_add", 0.0)

    def I(self, x):
        # NOTE: we don't use `p_yGx_sample` because calls to `p_yGx` take the f(x) to
        # prevent extra encoder forward passes.
        bs = tf.shape(x)[0]
        fx = self.enc.f(x)
        y = fx + self.enc.p_eps_sample(n=bs)
        # [[f(x1), ..., f(xN)], ..., [f(x1), ..., f(xN)]]
        fx_tile = tf.tile(tf.expand_dims(fx, 0), [bs, 1, 1])
        # [[y1, ..., y1], ..., [yN, ..., yN]]
        y_tile = tf.tile(tf.expand_dims(y, 1), [1, bs, 1])
        # [[p(y1|x1), ..., p(y1|xN)], ..., [p(yN|x1), ..., p(yN|xN)]]
        conditionals = self.enc.p_yGx(y_tile, fx_tile)
        # [p(y1|x1), ..., p(yN|xN)]
        d = tf.linalg.diag_part(conditionals)
        # [sum_j≠i p(y1|xj), ..., sum_j≠i p(yN|xj)]
        m_unnorm = tf.reduce_sum(conditionals, axis=1) - d
        m = m_unnorm / tf.cast(bs - 1, tf.float32)
        return tf.reduce_mean(tf.math.log(d / (m + self.hp.div_add)))
