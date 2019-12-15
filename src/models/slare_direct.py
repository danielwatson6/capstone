import time

import tensorflow as tf

import boilerplate as tfbp
from models import slare


@tfbp.default_export
class SLARE_Direct(slare):
    """SLARE model with direct, sampling-based entropy estimation."""

    default_hparams = {
        **slare.default_hparams,
    }

    def p_Y(self, x):
        """PDF estimate of Y = f(X) + E."""
        # The second half of the batch is interpreted as auxiliary negative examples, and
        # p_Y is computed for the first half of the batch.
        x_pos, x_neg = tf.split(x, 2)
        e_pos = self.P_E.sample(tf.shape(x_pos)[0])
        y_pos = self.f(x_pos) + e_pos
        return tf.reduce_mean(self.p_E(y_pos - self.f(x_neg)), axis=1)

    @tfbp.runnable
    def train(self, data_loader):
        train_data, test_data = data_loader.load()

        if self.hparams.optimizer == "adam":
            opt = tf.optimizers.Adam(self.hparams.lr)
        else:
            opt = tf.optimizers.SGD(self.hparams.lr)

        tmi = self.hparams.target_mi

        for epoch in range(self.hparams.epochs):
            print("Epoch", epoch)
            step = 0
            for x, _ in train_data:
                t0 = time.time()
                with tf.GradientTape() as g:
                    estimated_mi, loss = self.loss(x)
                grads = g.gradient(loss, self.trainable_weights)
                opt.apply_gradients(zip(grads, self.trainable_weights))

                t1 = time.time()
                self.maybe_print_stats(step, t1 - t0, estimated_mi)
                step += 1

        self.save()
