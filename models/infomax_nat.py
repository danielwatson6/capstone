import researchflow as rf
import tensorflow as tf

from models import infomax as InfoMax


input_signature = (
    tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    tf.TensorSpec(shape=[None, None], dtype=tf.float32),
)


def half_tanh(x):
    return 0.5 * tf.math.tanh(x)


@rf.export
class InfoMaxAE(InfoMax):
    """Autoencoder."""

    @staticmethod
    def hparams(hp):
        InfoMax.hparams(hp)
        hp.Fixed("dec_hidden_size", 1024)
        hp.Fixed("dec_num_hidden", 2)
        hp.Choice("dec_activation", ["tanh", "relu"], default="tanh")

    def loss(self, xy):
        x, y_target = xy
        y = self.enc.p_yGx_sample(x)
        return tf.reduce_mean((y - y_target) ** 2)

    @tf.function(input_signature=input_signature)
    def train_step(self, xy):
        with tf.GradientTape() as g:
            loss = self.loss(xy)

        grads = g.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))
        return loss, -loss

    def valid_step(self, x):
        loss = self.I(x)
        return loss, -loss