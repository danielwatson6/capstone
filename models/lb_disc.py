import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
from models import mi_disc as MIDisc


@tfbp.default_export
class LBDisc(MIDisc):
    """Discriminator-based MI lower bounder boilerplate."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.T.add(tfkl.Dense(1, activation=tf.math.softplus))

    def I(self, x):
        y = self.enc.p_yGx_sample(x)
        xy = tf.concat([x, y], axis=1)
        x_y = tf.concat([tf.random.shuffle(x), y], axis=1)
        return self.I_pos(xy), self.I_neg(x_y)

    def train_step_fixed_enc(self, x):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(self.T.trainable_weights)
            mi_split = self.I(x)
            mi = tf.reduce_sum(mi_split)
            if self.hp.disc_iter:
                loss_this_step = -mi_split[self.step % 2]
            else:
                loss_this_step = -mi

        grads = g.gradient(loss_this_step, self.T.trainable_weights)
        self.disc_opt.apply_gradients(zip(grads, self.T.trainable_weights))
        return -mi, mi

    def train_step(self, x):
        with tf.GradientTape(persistent=True) as g:
            mi_split = self.I(x)
            enc_loss = -tf.reduce_sum(mi_split)
            if self.hp.disc_iter:
                disc_loss = -mi_split[self.step % 2]
            else:
                disc_loss = enc_loss

        enc_grads = g.gradient(enc_loss, self.enc.trainable_weights)
        disc_grads = g.gradient(disc_loss, self.T.trainable_weights)
        del g
        self.opt.apply_gradients(zip(enc_grads, self.enc.trainable_weights))
        self.disc_opt.apply_gradients(zip(disc_grads, self.T.trainable_weights))
        return enc_loss, -enc_loss

    def valid_step(self, x):
        mi = tf.reduce_sum(self.I(x))
        return -mi, mi
