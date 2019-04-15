import tensorflow as tf
from .features import quantiles as qfs
from .com.enums import STANDARD_KERNEL


def QMCF_BBQ(bqp_points, x_query, qf):
    """

    :param bqp_points:
    :param x_query: (D, M)
                    The matrix of values to elementwise qf() transform
    :param qf:  function
                the interpolating function
    :param tfdt:
    :return:
    """
    S = qf(bqp_points[:, 0], bqp_points[:, 1], x_query)
    return tf.transpose(S)


def QMCF(sequence, kernel_type=STANDARD_KERNEL.RBF, tfdt=tf.float64,
         name="samplers.QMCF"):
    """
    General wrapper for all standard Quasi-Monte Carlo (Fourier) Features
    :param sequence:     QMCSequence object
    :param kernel_type:  enum
    :param tfdt:         tf.dtype
    :param name:         string
    :return:
    """
    quantiles_LUT = {STANDARD_KERNEL.RBF: qfs.standard_normal,
                     STANDARD_KERNEL.M12: qfs.matern_12,
                     STANDARD_KERNEL.M32: qfs.matern_32,
                     STANDARD_KERNEL.M52: qfs.matern_52}
    qf = quantiles_LUT[kernel_type]
    S = tf.Variable(initial_value=qf(sequence.points, tfdt=tfdt),
                    dtype=tfdt,
                    trainable=False,
                    name=name)
    return tf.transpose(S)


def RFF_RBF(D, m, tfdt=tf.float64, name="samplers.RFF_RBF"):
    """
    We use the full [cos(wx),sin(wx)] decomposition here as opposed to R&R's sqrt(2)cos(wx+b)
    This essentially samples the weights from the standard normal distribution N~(0,1)
    :param D:
    :param m:
    :param tfdt:    tensorflow datatype
                        The return type of the sampled variable
    :return: list of Tensorflow Tensor ops
    """
    S = tf.Variable(initial_value=tf.random_normal(shape=[D, m], dtype=tfdt),
                    trainable=False,
                    dtype=tfdt,
                    name=name)
    return S
