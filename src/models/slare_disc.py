import time

import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp
from models import slare
from utils import get_norms, tanh_squeezed


@tfbp.default_export
class SLARE_Disc(slare):
    """Abstract SLARE model with a discriminator."""

    default_hparams = {
        **slare.default_hparams,
        "disc": "dv",  # dv, gan, ganhack, fdiv
        "disc_hidden_sizes": [256],
        "disc_separate": False,
        "disc_lr": 0.1,
        "disc_optimizer": "sgd",  # sgd or adam
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        self.T = tf.keras.Sequential()
        for size in self.hparams.hidden_sizes:
            self.T.add(tfkl.Dense(size, activation=tf.math.tanh))

        activation = tf.math.softplus
        if self.hparams.disc in ["gan", "ganhack"]:
            activation = tanh_squeezed
        self.T.add(tfkl.Dense(1, activation=activation))

    def T_inputs(self, x, y):
        """Build inputs for the discriminator."""
        raise NotImplementedError

    def loss_inputs(self, x, y):
        """Build inputs for the main loss."""
        raise NotImplementedError

    def T_ratio(self, inputs):
        """PDF ratio estimate as a transformation of the discriminator."""
        t = self.T(inputs)
        if self.hparams.disc == "gan":
            return t * tf.math.reciprocal(1.0 - t)
        return t

    def loss_pos(self, inputs):
        """Discriminator loss for positive examples."""
        return -tf.reduce_mean(tf.math.log(self.T(inputs)))

    def loss_neg(self, inputs):
        """Discriminator loss for negative examples."""
        if self.hparams.disc == "gan":
            return -tf.reduce_mean(tf.math.log(1.0 - self.T(inputs)))
        if self.hparams.disc == "ganhack":
            return tf.reduce_mean(tf.math.log(self.T(inputs)))
        if self.hparams.disc == "fdiv":
            return tf.reduce_mean(tf.math.exp(-1.0) * self.T(inputs))
        return tf.math.log(tf.reduce_mean(self.T(inputs)))

    def loss_disc(self, inputs_pos, inputs_neg):
        """Total loss of the discriminator."""
        return self.loss_pos(inputs_pos) + self.loss_neg(inputs_neg)

    @tfbp.runnable
    def train(self, data_loader):
        train_data, valid_data = data_loader.load()

        if self.hparams.optimizer == "adam":
            opt = tf.optimizers.Adam(self.hparams.lr)
        else:
            opt = tf.optimizers.SGD(self.hparams.lr)

        if self.hparams.disc_optimizer == "adam":
            opt_disc = tf.optimizers.Adam(self.hparams.disc_lr)
        else:
            opt_disc = tf.optimizers.SGD(self.hparams.disc_lr)

        step = 0
        for epoch in range(self.hparams.epochs):
            print("Epoch", epoch)
            for x, _ in train_data:

                # Ensure to build the weights of the model.
                if step == 0:
                    self.T(self.T_inputs(x, self.f(x))[0])

                t0 = time.time()

                with tf.GradientTape(watch_accessed_variables=False) as g:
                    g.watch(self.f.trainable_weights)
                    y = self.f(x) + self.P_E.sample(tf.shape(x)[0])
                    emi, loss = self.loss(self.loss_inputs(x, y))

                with tf.GradientTape(watch_accessed_variables=False) as gg:
                    gg.watch(self.T.trainable_weights)
                    pos, neg = self.T_inputs(x, y)

                    if self.hparams.disc_separate:
                        if step % 2 == 0:
                            loss_disc = self.loss_pos(pos)
                        else:
                            loss_disc = self.loss_neg(neg)
                    else:
                        loss_disc = self.loss_disc(pos, neg)

                grads = g.gradient(loss, self.f.trainable_weights)
                opt.apply_gradients(zip(grads, self.f.trainable_weights))

                grads_disc = gg.gradient(loss_disc, self.T.trainable_weights)
                opt_disc.apply_gradients(zip(grads_disc, self.T.trainable_weights))

                t1 = time.time()

                if step % self.hparams.num_steps_per_eval == 0:
                    # TEMPORARY.
                    with tf.GradientTape(
                        persistent=True, watch_accessed_variables=False
                    ) as g:
                        g.watch(self.f.trainable_weights)
                        y = self.f(x)
                        p, n = self.T_inputs(x, y)
                        t = self.T(p)  # the gradients die here
                    print(
                        get_norms(
                            g.gradient(y, self.f.trainable_weights),
                            self.f.trainable_weights,
                        )
                    )
                    print(
                        get_norms(
                            g.gradient(t, self.f.trainable_weights),
                            self.f.trainable_weights,
                        )
                    )
                    print(tf.reduce_mean(t))

                    # Train stats.
                    self.print_stats(step, t1 - t0, emi)
                    if self.hparams.disc_separate:
                        loss_disc = self.loss_disc(pos, neg)

                    with self.train_writer.as_default():
                        tf.summary.scalar("loss", -loss, step=step)
                        tf.summary.scalar("loss_disc", -loss_disc, step=step)
                        tf.summary.scalar("estimated_mi", emi, step=step)
                        tf.summary.scalar("cross_entropy", emi + self.H_E(), step=step)
                        tf.summary.scalar(
                            "main_grad_norm", tf.linalg.global_norm(grads), step=step
                        )
                        tf.summary.scalar(
                            "disc_grad_norm",
                            tf.linalg.global_norm(grads_disc),
                            step=step,
                        )

                    # Validation stats.
                    x, _ = next(valid_data)
                    y = self.f(x) + self.P_E.sample(tf.shape(x)[0])
                    emi, loss = self.loss(self.loss_inputs(x, y))
                    loss_disc = self.loss_disc(*self.T_inputs(x, y))
                    self.print_stats(step, t1 - t0, emi, valid=True)
                    with self.valid_writer.as_default():
                        tf.summary.scalar("loss", -loss, step=step)
                        tf.summary.scalar("loss_disc", -loss_disc, step=step)
                        tf.summary.scalar("estimated_mi", emi, step=step)
                        tf.summary.scalar("cross_entropy", emi + self.H_E(), step=step)

                step += 1

        self.save()
