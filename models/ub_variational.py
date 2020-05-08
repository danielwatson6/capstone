import numpy as np
import researchflow as rf
import tensorflow as tf

from models import mi as MI


input_signature = (tf.TensorSpec(shape=[None, None], dtype=tf.float32),)


@rf.export
class UBVariational(MI):
    """Variational MI upper bounder."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.logvar = tf.Variable(tf.random.normal([self.hp.latent_size]))
        self.mean = tf.Variable(tf.zeros([self.hp.latent_size]))

    def log_q_y(self, y):
        """Evaluate log(q(y))."""
        return -0.5 * (
            tf.reduce_sum(tf.math.exp(self.logvar) * (y - self.mean) ** 2, axis=-1)
            + self.hp.latent_size * np.log(2 * np.pi)
            + tf.reduce_sum(self.logvar, axis=-1)
        )

    @tf.function(input_signature=input_signature)
    def I(self, x, y=None, ce=None):
        if y is None and ce is None:
            y = self.enc.p_yGx_sample(x)
        if ce is None:
            ce = -tf.reduce_mean(self.log_q_y(y))
        return ce - self.enc.H_eps

    @tf.function(input_signature=input_signature)
    def train_step_fixed_enc(self, x):
        weights = [self.mean, self.logvar]
        y = self.enc.p_yGx_sample(x)

        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(weights)
            ce = -tf.reduce_mean(self.log_q_y(y))

        grads = g.gradient(ce, weights)
        self.opt.apply_gradients(zip(grads, weights))

        return ce, self.I(x, ce=ce)

    @tf.function(input_signature=input_signature)
    def valid_step(self, x):
        y = self.enc.p_yGx_sample(x)
        ce = -tf.reduce_mean(self.log_q_y(y))
        return ce, self.I(x, ce=ce)
