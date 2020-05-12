import researchflow as rf
import tensorflow as tf
import tensorflow.keras.layers as tfkl

from models import infomax as InfoMax


input_signature = (tf.TensorSpec(shape=[None, None], dtype=tf.float32),)


def shifted_tanh(x):
    return tf.math.sigmoid(2.0 * x)


@rf.export
class InfoMaxGAN(InfoMax):
    """Jensen-Shannon-divergence-based MI maximizer."""

    @staticmethod
    def hparams(hp):
        InfoMax.hparams(hp)
        hp.Fixed("disc_hidden_size", 1024)
        hp.Fixed("disc_num_hidden", 2)
        hp.Choice("disc_activation", ["tanh", "relu"], default="tanh")
        hp.Boolean("use_nlog", default=False)
        hp.Boolean("disc_iter", default=False)
        hp.Float("disc_lr", 1e-3, 0.5, default=0.1, sampling="log")
        hp.Fixed("disc_clipnorm", 10.0)

    def __init__(self, **kw):
        super().__init__(**kw)

        # Discriminator neural network.
        self.T = tf.keras.Sequential()
        for _ in range(self.hp.disc_num_hidden):
            self.T.add(
                tfkl.Dense(self.hp.disc_hidden_size, activation=self.hp.disc_activation)
            )
        self.T.add(tfkl.Dense(1, activation=shifted_tanh))

        # Optimizer.
        self.disc_opt = tf.optimizers.SGD(
            self.hp.disc_lr, clipnorm=self.hp.disc_clipnorm
        )

    def loss_parts(self, x):
        y = self.enc.p_yGx_sample(x)
        y_random = tf.random.uniform(tf.shape(y), minval=-0.5, maxval=0.5)
        if self.hp.use_nlog:
            log_y_random = -tf.math.log(self.T(y_random))
        else:
            log_y_random = tf.math.log(1.0 - self.T(y_random))
        return tf.reduce_mean(tf.math.log(self.T(y))) + tf.reduce_mean(log_y_random)

    def loss(self, x):
        parts = self.loss_parts(x)
        return parts[0] + parts[1]

    @tf.function(input_signature=input_signature)
    def train_step(self, x):
        with tf.GradientTape(persistent=True):
            parts = self.loss_parts(x)
            enc_loss = parts[0] + parts[1]
            if self.hp.disc_iter:
                if self.step % 2 == 0:
                    disc_loss = -parts[0]
                else:
                    disc_loss = -parts[1]
            else:
                disc_loss = -enc_loss

        g_disc = g.gradient(disc_loss, self.T.trainable_weights)
        self.disc_opt.apply_gradients(zip(g_disc, self.T.trainable_weights))
        g_enc = g.gradient(enc_loss, self.enc.trainable_weights)
        self.opt.apply_gradients(zip(g_enc, self.T.trainable_weights))
        del g
        return enc_loss, -enc_loss

    def valid_step(self, x):
        loss = self.loss(x)
        return loss, -loss
