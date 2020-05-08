import researchflow as rf
import tensorflow as tf
import tensorflow.keras.layers as tfkl

from models import mi as MI


@rf.export
class LB_AE(MI):
    """Autoencoder."""

    @staticmethod
    def hparams(hp):
        MI.hparams(hp)
        hp.Int("dec_hidden_size", 128, 1024, default=128, sampling="log")
        hp.Int("dec_num_hidden", 1, 3, default=1)
        hp.Choice("dec_activation", ["tanh", "relu"], default="tanh")
        hp.Choice("loss", ["square", "ce"], default="ce")

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
            self.dec.add(tfkl.Dense(28 * 28))

    def loss(self, x):
        logits = self.dec(self.enc.p_yGx_sample(x))
        if self.hp.loss == "square":
            return tf.reduce_mean((tf.math.sigmoid(logits) - x) ** 2)
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits)
        )

    @tf.function(input_signature=self.input_signature)
    def I(self, x):
        """Mutual information UP TO AN INTRACTABLE CONSTANT."""
        logits = self.dec(self.enc.p_yGx_sample(x))
        return -tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits)
        )

    @tf.function(input_signature=self.input_signature)
    def train_step_fixed_enc(self, x):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(self.dec.trainable_weights)
            loss = self.loss(x)

        grads = g.gradient(loss, self.dec.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.dec.trainable_weights))
        return loss, -loss

    @tf.function(input_signature=self.input_signature)
    def train_step(self, x):
        with tf.GradientTape() as g:
            loss = self.loss(x)

        grads = g.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))
        return loss, self.I(x)

    @tf.function(input_signature=self.input_signature)
    def valid_step(self, x):
        return self.loss(x), self.I(x)
