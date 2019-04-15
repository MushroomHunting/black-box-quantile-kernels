import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_actual, y_pred):
    return np.sqrt(mean_squared_error(y_actual, y_pred))


def mse(y_actual, y_pred):
    return mean_squared_error(y_actual, y_pred)


def smse(y_actual, y_pred):
    """
    Standardised mean squared error.

    Parameters
    ----------
    y_true: ndarray
        vector of true targets
    y_pred: ndarray
        vector of predicted targets

    Returns
    -------
    float:
        SMSE of predictions vs truth (scalar)

    Example
    -------
    >>> y_true = np.random.randn(100)
    >>> smse(y_actual, y_true)
    0.0
    >>> smse(y_actual, np.random.randn(100)) >= 1.0
    True
    """

    N = y_actual.shape[0]
    return ((y_actual - y_pred) ** 2).sum() / (N * y_actual.var())


def mnll(actual_mean, pred_mean, pred_var):
    """
    Mean Negative Log Likelihood
    :param y_actual:
    :param y_pred:
    :param pred_var:
    :return:
    """
    log_part = np.log(2 * np.pi * pred_var) / 2.0
    unc_part = np.square(pred_mean - actual_mean) / (2 * pred_var)
    summed_parts = log_part + unc_part
    MNLL = np.mean(summed_parts)
    return MNLL

def frobenius_norm(A, normalize_with=None):
   """
   Calculates the frobenius norm of a matrix
   Allows for a normalizing matrix B of which the frobenius norm is taken
   :param A:
   :param normalize_with:
   :return:
   """
   if normalize_with is None:
      return np.linalg.norm(A, ord="fro")
   else:
      return np.linalg.norm(A, ord="fro") / np.linalg.norm(normalize_with, ord="fro")
