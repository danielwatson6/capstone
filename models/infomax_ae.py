import researchflow as rf
import tensorflow as tf
import tensorflow.keras.layers as tfkl

from models import mi as MI


input_signature = (tf.TensorSpec(shape=[None, None], dtype=tf.float32),)


def shifted_tanh(x):
    return tf.math.sigmoid(2.0 * x)


@rf.export
class InfoMaxAE(MI):
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
        if self.hp.gaussian:
            raise ValueError("LB_AE does not support a Gaussian encoder.")
        else:
            self.dec.add(tfkl.Dense(28 * 28, activation=shifted_tanh))

    def I(self, x):
        """Mutual information UP TO AN INTRACTABLE CONSTANT."""
        y_reconstructed = self.dec(self.enc.p_yGx_sample(x))
        return tf.reduce_mean((y_reconstructed - x) ** 2)

    @tf.function(input_signature=input_signature)
    def train_step_fixed_enc(self, x):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(self.dec.trainable_weights)
            loss = self.I(x)

        grads = g.gradient(loss, self.dec.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.dec.trainable_weights))
        return loss, loss

    @tf.function(input_signature=input_signature)
    def train_step(self, x):
        with tf.GradientTape() as g:
            loss = self.I(x)

        grads = g.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))
        return loss, loss

    def valid_step(self, x):
        loss = self.I(x)
        return loss, loss
