import os
import sys

import tensorflow as tf

import boilerplate as tfbp
from models import encoder as Encoder


@tfbp.default_export
class MI(tfbp.Model):
    """MI bounder boilerplate."""

    default_hparams = {
        **Encoder.default_hparams,
        "opt": "adam",
        "lr": 1e-3,
        "epochs": 5,
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.step = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.epoch = tf.Variable(0, dtype=tf.int64, trainable=False)

        self.enc = Encoder(
            save_dir=os.path.join(kw["save_dir"], "..", "encoder"),
            enc_hidden=self.hp.enc_hidden,
            latent_size=self.hp.latent_size,
            var_eps=self.hp.var_eps,
            gaussian=self.hp.gaussian,
        )
        self._fix_enc = self.enc.is_saved()

        # Optimizer.
        if self.hp.opt.lower() == "adam":
            self.opt = tf.optimizers.Adam(self.hp.lr)
        else:
            self.opt = tf.optimizers.SGD(self.hp.lr)

    def I(self, x):
        """Get the estimated MI for a batch."""
        raise NotImplementedError

    def train_step_fixed_enc(self, x):
        """Train step for the MI bounder keeping the encoder fixed.

        Not supported by all models; e.g., LB_NCE and UB_LOO don't require training.
        """
        raise NotImplementedError

    def train_step(self, x):
        """Train step for the MI bounder and the encoder.

        Should only be overriden by lower bound models; it makes no sense to maximize MI
        by maximizing an upper bound.
        """
        raise NotImplementedError

    def valid_step(self, x):
        """Validation step."""
        raise NotImplementedError

    def save(self):
        super().save()
        if not self._fix_enc:
            self.enc.save()

    @tfbp.runnable
    # @tf.function
    def train(self, data_loader):
        if self.hp.gaussian and not self._fix_enc:
            return self.enc.save()

        ds_train, ds_valid = data_loader()
        train_writer = self.make_summary_writer("train")
        valid_writer = self.make_summary_writer("valid")

        # Build the model's weights.
        self.I(next(ds_valid))

        while self.epoch < self.hp.epochs:
            for x in ds_train:
                if self._fix_enc:
                    train_loss, train_mi = self.train_step_fixed_enc(x)
                else:
                    train_loss, train_mi = self.train_step(x)

                if self.step % 100 == 0:
                    valid_loss, valid_mi = self.valid_step(next(ds_valid))

                    with train_writer.as_default():
                        tf.summary.scalar("loss", train_loss, step=self.step)
                        tf.summary.scalar("mi", train_mi, step=self.step)

                    with valid_writer.as_default():
                        tf.summary.scalar("loss", valid_loss, step=self.step)
                        tf.summary.scalar("mi", valid_mi, step=self.step)

                self.step.assign_add(1)
            self.epoch.assign_add(1)
            self.save()

    @tfbp.runnable
    def estimate(self, data_loader):
        ds = data_loader()

        # Running average.
        avg = 0.0
        for i, batch in enumerate(ds, 1):
            # The `reduce_sum` call handles the case where `I` is a tuple.
            avg += (tf.reduce_sum(self.I(batch)) - avg) / i

        tf.print(avg, output_stream=sys.stdout)

    @tfbp.runnable
    def estimate_test(self, data_loader):
        return self.estimate(data_loader)
