import scipy.stats as sstats
import numpy as np


def norm(points, loc, scale):
    """
    Standard Normal
    :param scale:
    :param points:
    :param loc:
    :return:
    """
    return sstats.norm.ppf(points, loc, scale)


def standard_normal(points):
    """
    Standard Normal
    :param points:
    :return:
    """
    return sstats.norm.ppf(points)


def matern_12(points):
    """
    Matern 12
    Ref: Sampling Student''s T distribution-use of the inverse cumulative
    distribution function
    :param points:
    :return:
    """
    return np.tan(np.pi * (points - 0.5))


def matern_32(points):
    """
    Matern 32
    ref: Ref: Sampling Student''s T distribution-use of the inverse cumulative
    distribution function
    :param points:
    :return:
    """
    return (2 * points - 1) / np.sqrt(2 * points * (1 - points))


def matern_52(points):
    """
    Matern 52
    Ref: Sampling Student''s T distribution-use of the inverse cumulative
    distribution function
    :param points:
    :return:
    """
    alpha = 4 * points * (1 - points)
    p = 4 / np.sqrt(alpha) * np.cos((1 / 3) * np.arccos(np.sqrt(alpha)))
    return np.sign(points - 0.5) * np.sqrt(p - 4)
