import os.path
import sys

import researchflow as rf
import tensorflow as tf

from models import encoder as Encoder


@rf.export
class MI(rf.Model):
    """MI bounder boilerplate."""

    @staticmethod
    def hparams(hp):
        Encoder.hparams(hp)
        hp.Fixed("encoder", "")
        hp.Fixed("epochs", 10)
        hp.Choice("opt", ["sgd", "adam"], default="sgd")
        hp.Float("lr", 5e-4, 0.1, default=1e-3, sampling="log")
        hp.Fixed("clipnorm", 10.0)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.step = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.epoch = tf.Variable(0, dtype=tf.int64, trainable=False)

        encoder_save_dir = os.path.join(self.save_dir, "encoder")
        if self.hp.encoder:
            encoder_save_dir = os.path.join(self.save_dir, self.hp.encoder)
        self.enc = Encoder(
            save_dir=os.path.join(encoder_save_dir), hparams=kw.get("hparams"),
        )
        self._fix_enc = self.enc.is_saved()

        # Optimizer.
        if self.hp.opt == "adam":
            self.opt = tf.optimizers.Adam(self.hp.lr, clipnorm=self.hp.clipnorm)
        else:
            self.opt = tf.optimizers.SGD(self.hp.lr, clipnorm=self.hp.clipnorm)

        self.input_signature = (tf.TensorSpec(shape=[None, None], dtype=tf.float32),)

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

    @rf.cli
    def train(self, data_loader):
        tf.print(
            f"Training {self.__class__.__name__} model (fixed_encoder={self._fix_enc})"
        )
        tf.print("Saved at:", self.save_dir)
        tf.print("Hyperparameters:")
        for name, value in self.hp._asdict().items():
            tf.print(f"  {name}: {value}")

        ds_train, ds_valid = data_loader()
        self._train_writer = self.make_summary_writer("train")
        self._valid_writer = self.make_summary_writer("valid")

        # Build the model's weights.
        self.valid_step(next(ds_valid))

        while self.epoch < self.hp.epochs:
            for x in ds_train:
                t0 = tf.timestamp()
                if self._fix_enc:
                    train_loss, train_mi = self.train_step_fixed_enc(x)
                else:
                    train_loss, train_mi = self.train_step(x)
                t1 = tf.timestamp()

                if self.step % 100 == 0:
                    valid_loss, valid_mi = self.valid_step(next(ds_valid))
                    tf.print("step", self.step)
                    tf.print("  train step time:", t1 - t0)
                    tf.print("  valid mi:", valid_mi)

                    with self._train_writer.as_default():
                        tf.summary.scalar("loss", train_loss, step=self.step)
                        tf.summary.scalar("mi", train_mi, step=self.step)

                    with self._valid_writer.as_default():
                        tf.summary.scalar("loss", valid_loss, step=self.step)
                        tf.summary.scalar("mi", valid_mi, step=self.step)

                self.step.assign_add(1)
            self.epoch.assign_add(1)
            self.save()

    @rf.cli
    def estimate(self, data_loader):
        ds = data_loader()
        tf.print("evaluating at step", self.step)

        # Running average.
        avg = 0.0
        for i, batch in enumerate(ds, 1):
            # The `reduce_sum` call handles the case where `I` is a tuple.
            avg += (tf.reduce_sum(self.I(batch)) - avg) / i

        tf.print(avg, output_stream=sys.stdout)
        return avg

    @rf.cli
    def estimate_test(self, data_loader):
        return self.estimate(data_loader)

    @rf.cli
    def create(self, data_loader):
        ds = iter(data_loader())
        self.valid_step(next(ds))  # build the model's weights
        self.save()
