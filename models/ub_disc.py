import numpy as np
import tensorflow as tf

import boilerplate as tfbp
from models import mi_disc as MIDisc


@tfbp.default_export
class UBDisc(MIDisc):
    """Discriminator-based MI upper bounder boilerplate."""

    default_hparams = {
        **MIDisc.default_hparams,
        "var_xi": 1.0,
    }

    def p_xi_sample(self, n=1):
        """Sample from P_ξ."""
        shape = [n, self.enc.hp.latent_size]
        return tf.random.normal(shape, stddev=self.hp.var_xi ** 0.5)

    def p_xi(self, y):
        """Evaluate p_ξ(y)."""
        z = (2 * np.pi * self.hp.var_xi) ** (self.enc.hp.latent_size / 2)
        return (
            tf.math.exp(-tf.reduce_sum(y ** 2, axis=-1) / (2 * self.enc.hp.latent_size))
            / z
        )

    def q_y(self, y):
        """Evaluate the approximation to p_Y.

        This should be overriden depending on the discriminator's global optimum.
        """
        raise NotImplementedError

    def I(self, x, y=None):
        if y is None:
            y = self.enc.p_yGx_sample(x)
        return -tf.reduce_mean(tf.math.log(self.q_y(y))) - self.enc.H_eps

    def train_step_fixed_enc(self, x):
        y = self.enc.p_yGx_sample(x)
        xi = self.p_xi_sample(n=tf.shape(x)[0])

        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(self.T.trainable_weights)

            loss_pos = -self.I_pos(y)
            loss_neg = -self.I_neg(xi)
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
        xi = self.p_xi_sample(n=tf.shape(x)[0])
        return -self.I_pos(y) - self.I_neg(xi), self.I(x, y=y)
