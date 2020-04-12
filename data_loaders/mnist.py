import tensorflow as tf
import tensorflow_datasets as tfds

import boilerplate as tfbp


@tfbp.default_export
class MNIST(tfbp.DataLoader):
    default_hparams = {
        "batch_size": 32,
        "num_valid": 10000,
        "macro": False,
    }

    def __call__(self):
        ds_notest, ds_test = tfds.load("mnist", split=["train", "test"])

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
        ds = ds.map(
            lambda x: tf.reshape(tf.cast(x["image"], tf.float32) / 255.0, [-1, 28 * 28])
        )
        return ds.prefetch(1)
