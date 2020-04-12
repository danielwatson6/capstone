import tensorflow as tf

import boilerplate as tfbp


@tfbp.default_export
class Gaussian(tfbp.DataLoader):
    default_hparams = {
        "batch_size": 32,
        "latent_size": 10,
        "num_train": 50000,
        "num_valid": 10000,
        "macro": False,
    }

    def __call__(self):
        tf.random.set_seed(2020)
        ds_train = self._prep_dataset(self.hp.num_train)
        ds_valid = self._prep_dataset(self.hp.num_valid)

        if self.method == "train":
            return ds_train, iter(ds_valid.repeat())

        return ds_valid

    def _prep_dataset(self, n):
        ds = tf.data.Dataset.from_tensor_slices(
            tf.random.normal([n, self.hp.latent_size])
        )
        ds = ds.shuffle(10000)
        if self.hp.macro:
            ds = ds.batch(self.hp.num_valid)
        else:
            ds = ds.batch(self.hp.batch_size)
        return ds.prefetch(1)
