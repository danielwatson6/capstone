import researchflow as rf
import tensorflow as tf

from models import mi as MI


input_signature = (tf.TensorSpec(shape=[None, None], dtype=tf.float32),)


@rf.export
class LB_NCE(MI):
    """Noise-contrastive estimation MI lower bounder."""

    @tf.function(input_signature=input_signature)
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
        # [mean_j p(y1|xj), ..., mean_j p(yN|xj)]
        m = tf.reduce_mean(conditionals, axis=1)
        return tf.reduce_mean(tf.math.log(d) - tf.math.log(m))

    @tf.function(input_signature=input_signature)
    def train_step(self, x):
        with tf.GradientTape() as g:
            loss = -self.I(x)

        grads = g.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))
        return loss, -loss

    @tf.function(input_signature=input_signature)
    def valid_step(self, x):
        mi = self.I(x)
        return -mi, mi
