import researchflow as rf
import tensorflow as tf

from models import mi as MI


@rf.export
class LB_NCE(MI):
    """Noise-contrastive estimation MI lower bounder."""

    def I(self, x):
        # NOTE: we don't use `p_yGx_sample` because calls to `p_yGx` take the f(x) to
        # prevent extra encoder forward passes.
        bs = tf.shape(x)[0]
        fx = self.enc.f(x)
        y = fx + self.enc.p_eps_sample(n=bs)
        # [[f(x1), ..., f(x1)], ..., [f(xN), ..., f(xN)]]
        fx_tile = tf.tile(tf.expand_dims(fx, 1), [1, bs, 1])
        # [[y1, ..., yN], ..., [y1, ..., yN]]
        y_tile = tf.tile(tf.expand_dims(y, 0), [bs, 1, 1])
        # [[p(y1|x1), ..., p(yN|x1)], ..., [p(y1|xN), ..., p(yN|xN)]]
        conditionals = self.enc.p_yGx(y_tile, fx_tile)
        # [p(y1|x1), ..., p(yN|xN)]
        d = tf.linalg.diag_part(conditionals)
        # [mean_j p(yj|x1), ..., mean_j p(yj|xN)]
        m = tf.reduce_mean(conditionals, axis=1)
        return tf.reduce_mean(tf.math.log(d) - tf.math.log(m))

    def train_step(self, x):
        with tf.GradientTape() as g:
            loss = -self.I(x)

        grads = g.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))
        return loss, -loss

    def valid_step(self, x):
        mi = self.I(x)
        return -mi, mi
