import tensorflow as tf


def get_norms(grads, tvars):
    """Compute the mean norms of all the grads corresponding to the trainable weights."""
    result = []
    for grad, tvar in zip(grads, tvars):
        norm = tf.math.sqrt(tf.reduce_sum(grad ** 2)).numpy()
        result.append((tvar.name, norm))
    return result


def mish(x):
    """Mish activation function (Misra, 2019).

    https://arxiv.org/abs/1908.08681
    """
    return x * tf.math.tanh(tf.math.softplus(x))


def tanh_restricted(beta):
    def f(x):
        return tf.math.tanh(x) * (0.5 - beta) + 0.5

    return f


def tanh_squeezed(x):
    return 0.5 * (tf.math.tanh(x) + 1)
