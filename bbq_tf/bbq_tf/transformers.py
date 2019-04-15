import tensorflow as tf


def linear(X, has_bias_term=True, tfdt=tf.float64):
    """
    Allows a linear regression model to model a linear trend
    Optionally allows a bias term
    :param X:   (N,D)
                Raw data
    :param has_bias_term:  boolean

    :return:
    """
    shape = tf.shape(input=X)
    if has_bias_term is True:
        return tf.concat(axis=1,
                         values=[tf.ones(shape=[shape[0], 1], dtype=tfdt), X])
    else:
        return X


def cos_sin(X, S, tfdt=tf.float64, m=None):
    """
    The full feature form
    :param X:   (N,D)
                Data to embed with features
    :param S:   (D,M)
                Sampled frequencies
    :return:
    """
    if not m:
        D, m = S.shape
    inside_calc = tf.matmul(X, S, name="cos_sin_inside_calc")
    return tf.divide(
        tf.concat(axis=1, values=[tf.cos(inside_calc), tf.sin(inside_calc)],
                  name="cos_sin_concat"),
        tf.sqrt(tf.cast(m, dtype=tfdt)))


def cos_sin_ard(X, S1, S2, tfdt=tf.float64, m=None):
    """
    ARD transformer for bbq. Applies different quantile samples to different dimensions
    Currently implemented only for 2D
    :param X:   (N,D)
                Data to embed with features
    :param S1:  Frequencies for the first dimension
    :param S2:  Frequencies for the second dimension
    :return:
    """
    if not m:
        D, m = S1.shape
    inside_calc = tf.matmul(tf.reshape(X[:, 0], (-1, 1)),
                            tf.reshape(S1[0, :], (1, -1))) + \
                  tf.matmul(tf.reshape(X[:, 1], (-1, 1)),
                            tf.reshape(S2[1, :], (1, -1)))
    return tf.divide(
        tf.concat(axis=1, values=[tf.cos(inside_calc),
                                  tf.sin(inside_calc)],
                  name="cos_sin_concat"),
        tf.sqrt(tf.cast(m, dtype=tfdt)))
