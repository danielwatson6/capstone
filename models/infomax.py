import researchflow as rf
import tensorflow as tf

from models import mi as MI


@rf.export
class InfoMax(MI):
    """InfoMax boilerplate."""

    def loss(self, x):
        raise NotImplementedError

    @rf.cli
    def evaluate(self, data_loader):
        ds = data_loader()
        tf.print("evaluating at step", self.step)

        # Running average.
        avg = 0.0
        for i, batch in enumerate(ds, 1):
            avg += (self.loss(batch) - avg) / i

        tf.print(avg, output_stream=sys.stdout)
        return avg
