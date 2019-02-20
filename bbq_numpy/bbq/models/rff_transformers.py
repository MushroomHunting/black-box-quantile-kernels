import numpy as np


def linear(x, bias_term=True):
    """
    Allows a linear regression model to model a linear trend
    Optionally allows a bias term
    :param x: Your source data matrix (typically this is on your raw data, e.g.
    time)
    :param bias_term:  boolean

    :return:
    """
    n, d = x.shape
    if bias_term is True:
        return np.hstack((np.ones((n, 1)), x))
    else:
        return x


def cos_sin(x, s):
    """
    The full form not used in the original Rahimi Recht 2007 paper
    Pairs with samplers.RFF_RBF
    :param x: (N,D) nd-array
    :param s:
    :return:
    """

    c = s.shape[1]
    inside_calc = np.dot(x, s)
    return np.hstack((np.cos(inside_calc), np.sin(inside_calc))) / np.sqrt(c)
