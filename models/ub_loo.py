import tensorflow as tf

import boilerplate as tfbp
from models import mi as MI


@tfbp.default_export
class UB_LOO(MI):
    """Leave-one-out MI upper bounder."""

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
        q_y_unnorm = tf.reduce_sum(conditionals, axis=1) - d
        q_y = q_y_unnorm / tf.cast(bs - 1, tf.float32)
        return -tf.reduce_mean(tf.math.log(q_y)) - self.enc.H_eps
