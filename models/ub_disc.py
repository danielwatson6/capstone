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

    def q_sample(self, n=1):
        """Sample from Q_Y."""
        shape = [n, self.enc.hp.latent_size]
        return tf.random.normal(shape, stddev=self.hp.var_q ** 0.5)

    def log_q(self, y):
        """Evaluate log(q(y))."""
        log_z = 0.5 * self.hp.latent_size * np.log(2 * np.pi * self.hp.var_q)
        return -tf.reduce_sum(y ** 2, axis=-1) / (2 * self.hp.var_q) - log_z

    def D_pos(self, y):
        """Compute the KL[P_Y||Q_Y] term for positive samples."""
        return tf.reduce_mean(tf.math.log(self.T(y)))

    def D_neg(self, y):
        """Compute the KL[P_Y||Q_Y] term for negative samples."""
        raise NotImplementedError

    def I(self, x, y=None, y_q=None, d=None):
        if y is None:
            y = self.enc.p_yGx_sample(x)
        if y_q is None and d is None:
            y_q = self.q_sample(n=tf.shape(x)[0])
        if d is None:
            d = self.D_pos(y) + self.D_neg(y_q)
        return -(tf.reduce_mean(self.log_q(y)) + self.enc.H_eps + d)

    def train_step_fixed_enc(self, x):
        y = self.enc.p_yGx_sample(x)
        y_q = self.q_sample(n=tf.shape(x)[0])

        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(self.T.trainable_weights)

            loss_pos = -self.D_pos(y)
            loss_neg = -self.D_neg(y_q)
            loss = loss_pos + loss_neg

            if self.hp.disc_iter and self.step % 2 == 0:
                loss_disc = loss_pos
            elif self.hp.disc_iter:
                loss_disc = loss_neg
            else:
                loss_disc = loss

        grads = g.gradient(loss_disc, self.T.trainable_weights)
        self.disc_opt.apply_gradients(zip(grads, self.T.trainable_weights))
        return loss, self.I(x, y=y, d=-loss)

    def valid_step(self, x):
        y = self.enc.p_yGx_sample(x)
        y_q = self.q_sample(n=tf.shape(x)[0])
        loss = -self.D_pos(y) - self.D_neg(y_q)
        return loss, self.I(x, y=y, d=-loss)
