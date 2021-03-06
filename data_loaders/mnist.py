import researchflow as rf
import tensorflow as tf
import tensorflow_datasets as tfds


@rf.export
class MNIST(rf.DataLoader):
    """MNIST data loader."""

    @staticmethod
    def hparams(hp):
        hp.Int("batch_size", 16, 512, default=32, sampling="log")
        hp.Fixed("num_valid", 10000)
        hp.Fixed("macro", False)

    def __call__(self):
        tf.random.set_seed(2020)
        ds_notest, ds_test = tfds.load(
            "mnist",
            split=["train", "test"],
            read_config=tfds.ReadConfig(try_autocache=False),
        )
        ds_valid = self._prep_dataset(ds_notest.take(self.hp.num_valid))
        ds_train = self._prep_dataset(ds_notest.skip(self.hp.num_valid))
        ds_test = self._prep_dataset(ds_test)

        if self.method == "train":
            return ds_train, iter(ds_valid.repeat())

        elif self.method == "estimate_test":
            return ds_test

        return ds_valid

    def _prep_dataset(self, ds):
        ds = ds.shuffle(10000)
        if self.hp.macro:
            ds = ds.batch(self.hp.num_valid)
        else:
            ds = ds.batch(self.hp.batch_size)
        # Flatten and normalize to [-.5, .5]
        ds = ds.map(
            lambda x: tf.reshape(
                tf.cast(x["image"], tf.float32) / 255.0 - 0.5, [-1, 28 * 28]
            )
        )
        return ds.prefetch(1)
