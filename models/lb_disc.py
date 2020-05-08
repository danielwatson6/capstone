import researchflow as rf
import tensorflow as tf
import tensorflow.keras.layers as tfkl

from models import mi_disc as MIDisc


@rf.export
class LBDisc(MIDisc):
    """Discriminator-based MI lower bounder boilerplate."""

    def I_pos(self, xy):
        """Compute the mutual information term for positive samples."""
        return tf.reduce_mean(tf.math.log(self.T(xy)))

    def I_neg(self, xy):
        """Compute the mutual information term for negative samples."""
        raise NotImplementedError

    def I(self, x):
        y = self.enc.p_yGx_sample(x)
        xy_joint = tf.concat([x, y], axis=1)
        xy_marginals = tf.concat([tf.random.shuffle(x), y], axis=1)
        return self.I_pos(xy_joint), self.I_neg(xy_marginals)

    def train_step_fixed_enc(self, x):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(self.T.trainable_weights)
            mi_split = self.I(x)
            mi = tf.reduce_sum(mi_split)
            if self.hp.disc_iter:
                if self.step % 2 == 0:
                    disc_loss = -mi_split[0]
                else:
                    disc_loss = -mi_split[1]
            else:
                disc_loss = -mi

        grads = g.gradient(disc_loss, self.T.trainable_weights)
        self.disc_opt.apply_gradients(zip(grads, self.T.trainable_weights))
        return -mi, mi

    def train_step(self, x):
        with tf.GradientTape(persistent=True) as g:
            mi_split = self.I(x)
            enc_loss = -tf.reduce_sum(mi_split)
            if self.hp.disc_iter:
                if self.step % 2 == 0:
                    disc_loss = -mi_split[0]
                else:
                    disc_loss = -mi_split[1]
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
