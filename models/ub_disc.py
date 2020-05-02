import numpy as np
import researchflow as rf
import tensorflow as tf

from models import mi_disc as MIDisc


@rf.export
class UBDisc(MIDisc):
    """Discriminator-based MI upper bounder boilerplate."""

    @staticmethod
    def hparams(hp):
        MIDisc.hparams(hp)
        hp.Float("var_q", 1e-3, 1.0, default=0.1, sampling="log")

    def q_y_sample(self, n=1):
        """Sample from Q_Y."""
        shape = [n, self.enc.hp.latent_size]
        return tf.random.normal(shape, stddev=self.hp.var_q ** 0.5)

    def q_y(self, y):
        """Evaluate q_Y."""
        z = (2 * np.pi * self.hp.var_q) ** (self.enc.hp.latent_size / 2)
        return (
            tf.math.exp(-tf.reduce_sum(y ** 2, axis=-1) / (2 * self.enc.hp.latent_size))
            / z
        )

    def I(self, x, y=None):
        if y is None:
            y = self.enc.p_yGx_sample(x)
        return ...  # TODO

    def train_step_fixed_enc(self, x):
        y = self.enc.p_yGx_sample(x)
        y_q = self.q_y_sample(n=tf.shape(x)[0])

        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(self.T.trainable_weights)

            loss_pos = -self.I_pos(y)
            loss_neg = -self.I_neg(y_q)
            loss = loss_pos + loss_neg

            if self.hp.disc_iter and self.step % 2 == 0:
                loss_this_step = loss_pos
            elif self.hp.disc_iter:
                loss_this_step = loss_neg
            else:
                loss_this_step = loss

        grads = g.gradient(loss_this_step, self.T.trainable_weights)
        self.disc_opt.apply_gradients(zip(grads, self.T.trainable_weights))
        return loss, self.I(x, y=y)

    def valid_step(self, x):
        y = self.enc.p_yGx_sample(x)
        y_q = self.q_y_sample(n=tf.shape(x)[0])
        return -self.I_pos(y) - self.I_neg(y_q), self.I(x, y=y)
