import scipy.stats as sstats
import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tfd


def norm(points, loc, scale, tfdt):
    """
    Standard Normal
    :param points:
    :param loc:     mean
    :param scale:   standard deviation
    :return:
    """
    _loc = tf.constant(loc, dtype=tfdt)
    _scale = tf.constant(scale, dtype=tfdt)
    p = tfd.Normal(loc=_loc, scale=_scale)
    return p.quantile(points)


def standard_normal(points, tfdt):
    """
    Standard Normal
    :param points:
    :return:
    """
    _loc = tf.constant(0.0, dtype=tfdt)
    _scale = tf.constant(1.0, dtype=tfdt)
    p = tfd.Normal(loc=_loc, scale=_scale)
    return p.quantile(points)


def matern_12(points):
    """
    Matern 12
    Ref: Sampling Student''s T distribution-use of the inverse cumulative distribution function
    :param points:
    :return:
    """
    return tf.tan(np.pi * (points - 0.5))


def matern_32(points):
    """
    Matern 32
    ref: Ref: Sampling Student''s T distribution-use of the inverse cumulative distribution function
    :param points:
    :return:
    """
    return (2 * points - 1) / tf.sqrt(2 * points * (1 - points))


def matern_52(points):
    """
    Matern 52
    Ref: Sampling Student''s T distribution-use of the inverse cumulative distribution function
    :param points:
    :return:
    """
    alpha = 4 * points * (1 - points)
    p = 4 / tf.sqrt(alpha) * tf.cos((1 / 3) * tf.acos(tf.sqrt(alpha)))
    return tf.sign(points - 0.5) * tf.sqrt(p - 4)
