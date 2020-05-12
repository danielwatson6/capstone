import researchflow as rf
import tensorflow as tf
import tensorflow_datasets as tfds


@rf.export
class BinarizedMNIST(rf.DataLoader):
    """MNIST data loader."""

    @staticmethod
    def hparams(hp):
        hp.Int("batch_size", 16, 512, default=32, sampling="log")
        hp.Fixed("num_valid", 10000)
        hp.Fixed("latent_size", 10)

    def __call__(self):
        tf.random.set_seed(2020)
        ds_notest, ds_test = tfds.load(
            "mnist",
            split=["train", "test"],
            read_config=tfds.ReadConfig(try_autocache=False),
        )
        ds_valid = ds_notest.take(self.hp.num_valid)

        if self.method == "train" or self.method.startswith("evaluate"):
            ds_train = ds_notest.skip(self.hp.num_valid)

            targets_train = ds = tf.data.Dataset.from_tensor_slices(
                tf.random.uniform(
                    [60000 - self.hp.num_valid, self.hp.latent_size],
                    minval=-0.5,
                    maxval=0.5,
                )
            )
            targets_valid = ds = tf.data.Dataset.from_tensor_slices(
                tf.random.uniform(
                    [self.hp.num_valid, self.hp.latent_size], minval=-0.5, maxval=0.5
                )
            )
            targets_test = ds = tf.data.Dataset.from_tensor_slices(
                tf.random.uniform([10000, self.hp.latent_size], minval=-0.5, maxval=0.5)
            )
            ds_train = self._prep_dataset_with_targets(ds_train, targets_train)
            ds_valid = self._prep_dataset_with_targets(ds_valid, targets_valid)
            ds_test = self._prep_dataset_with_targets(ds_test, targets_test)

            if self.method == "train":
                return ds_train, iter(ds_valid.repeat())
            if self.method == "evaluate_test":
                return ds_test
            return ds_valid

        ds_valid = self._prep_dataset(ds_valid)
        ds_test = self._prep_dataset(ds_test)

        if self.method == "estimate_test":
            return ds_test
        return ds_valid

    def _prep_dataset_with_targets(self, inputs, targets):
        ds = tf.data.Dataset.zip((inputs, targets))
        ds = ds.shuffle(10000)
        ds = ds.batch(self.hp.batch_size)
        ds = ds.map(
            lambda x: (
                tf.reshape(tf.cast(x[0]["image"], tf.float32) / 255.0, [-1, 28 * 28]),
                x[1],
            )
        )
        return ds.prefetch(1)

    def _prep_dataset(self, ds):
        ds = ds.shuffle(10000)
        ds = ds.batch(self.hp.batch_size)
        ds = ds.map(
            lambda x: tf.reshape(tf.cast(x["image"], tf.float32) / 255.0, [-1, 28 * 28])
        )
        return ds.prefetch(1)
