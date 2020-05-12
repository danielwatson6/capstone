import researchflow as rf
import tensorflow as tf
import tensorflow.keras.layers as tfkl

from models import infomax as InfoMax


input_signature = (tf.TensorSpec(shape=[None, None], dtype=tf.float32),)


def half_tanh(x):
    return 0.5 * tf.math.tanh(x)


@rf.export
class InfoMaxAE(InfoMax):
    """Autoencoder."""

    @staticmethod
    def hparams(hp):
        MI.hparams(hp)
        hp.Fixed("dec_hidden_size", 1024)
        hp.Fixed("dec_num_hidden", 2)
        hp.Choice("dec_activation", ["tanh", "relu"], default="tanh")

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        # Decoder neural network.
        self.dec = tf.keras.Sequential()
        for _ in range(self.hp.dec_num_hidden):
            self.dec.add(
                tfkl.Dense(self.hp.dec_hidden_size, activation=self.hp.dec_activation)
            )
        # TODO: abstract the output dim to the encoder's input dim.
        self.dec.add(tfkl.Dense(28 * 28, activation=half_tanh))

    def loss(self, x):
        y_reconstructed = self.dec(self.enc.p_yGx_sample(x))
        return tf.reduce_mean((y_reconstructed - x) ** 2)

    def I(self, x):
        """Mutual information lower bound UP TO AN INTRACTABLE CONSTANT.

        This is overriden because the reconstruction error is also used as a proxy MI
        evaluation, training a fresh decoder on a fixed encoder.
        """
        return -self.loss(x)

    # This is overriden because the reconstruction error is also used as a proxy MI
    # evaluation, training a fresh decoder on a fixed encoder.
    @tf.function(input_signature=input_signature)
    def train_step_fixed_enc(self, x):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(self.dec.trainable_weights)
            loss = self.loss(x)

        grads = g.gradient(loss, self.dec.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.dec.trainable_weights))
        return loss, -loss

    @tf.function(input_signature=input_signature)
    def train_step(self, x):
        with tf.GradientTape() as g:
            loss = self.loss(x)

        grads = g.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))
        return loss, -loss

    def valid_step(self, x):
        loss = self.I(x)
        return loss, -loss
