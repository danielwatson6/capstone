import tensorflow as tf

import boilerplate as tfbp


@tfbp.default_export
class MNIST(tfbp.DataLoader):
    default_hparams = {
        "batch_size": 32,
        "num_valid": 10000,
    }

    def load(self):
        train_valid_data, test_data = tf.keras.datasets.mnist.load_data()

        if self.method == "eval_test":
            test_data = tf.data.Dataset.from_tensor_slices(test_data)
            return self._transform_dataset(test_data)

        train_valid_data = tf.data.Dataset.from_tensor_slices(train_valid_data)
        valid_data = train_valid_data.take(self.hparams.num_valid)
        valid_data = self._transform_dataset(valid_data)

        if self.method == "train":
            train_data = train_valid_data.skip(self.hparams.num_valid)
            train_data = self._transform_dataset(train_data)
            return train_data, iter(valid_data.repeat())

        return valid_data

    def _transform_dataset(self, dataset):
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(self.hparams.batch_size)
        return dataset.map(
            lambda x, y: (
                tf.reshape(tf.cast(x, tf.float32), [-1, 28 * 28]),
                tf.cast(y, tf.int64),
            )
        )
