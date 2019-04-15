import numpy as np
import tensorflow as tf

def diff(a, axis=0):
    """
    equivalent of np.diff with n=1
    :param a:
    :param axis:
    :return:
    """
    if axis == 0:
        return a[1:] - a[:-1]
    elif axis == 1:
        return a[:, 1:] - a[:, :-1]

def smw_inv_correction(A_inv, U, V, tfdt=tf.float64):
    """
    Sherman-Morrison-Woodbury update
    For rank k updates to the inverse matrix

    IMPORTANT: This is the correction factor which one must assign.sub or subtract
    Usage:   tf.assign_sub() this value

    ref:     http://mathworld.wolfram.com/WoodburyFormula.html
             https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    :param A_inv:   n x n
    :param U:      n x k
    :param V:      k x n
    :param rank:   V.shape[0] (TENSOR 32 bit float)
    :return:
    """
    rank = tf.shape(V)[0]
    SU = tf.matmul(A_inv, U)
    VS = tf.matmul(V, A_inv)
    I_plus_VSU_inv = tf.matrix_inverse(
        input=tf.eye(rank, dtype=tfdt) + tf.matmul(VS, U))
    SU_I_plus_VSU = tf.matmul(SU, I_plus_VSU_inv)
    return tf.matmul(SU_I_plus_VSU, VS)


def smw_inv(A_inv, U, V, tfdt=tf.float64):
    """
    Sherman-Morrison-Woodbury update
    For rank k updates to the inverse matrix

    IMPORTANT: This is an in-place variable SMW update to A_inv

    ref:  http://mathworld.wolfram.com/WoodburyFormula.html
          https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    :param A_inv:  tf.Variable of shape n x n
    :param U:      n x k
    :param V:      k x n
    :return:
    """
    rank = tf.shape(V)[0]
    SU = tf.matmul(A_inv, U)
    VS = tf.matmul(V, A_inv)
    I_plus_VSU_inv = tf.matrix_inverse(
        input=tf.eye(rank, dtype=tfdt) + tf.matmul(VS, U))
    SU_I_plus_VSU = tf.matmul(SU, I_plus_VSU_inv)
    return tf.assign_sub(A_inv, tf.matmul(SU_I_plus_VSU, VS))


def cartesian_product(A, B, axis=1):
    """
    Pure tensorflow implementation that constsructs a row or column pairwise product for a pair of 2D tensors

    IMPORTANT: If B has a runtime size calcualtion, it CANNOT be argument A.
    Fix: Just put the 2 dimensional tensor in A
    Example: This will occur if you're multiplying a Nx2 Linear basis function with another one
             A = Linear, B = AnotherBasis   --> This is Ok
             A = AnotherBasis, B = Linear   --> This is not Ok
    Fix: Just put the Linear basis as the first item
    Example
    -------
    A = np.array([[1,2],
                  [3,4]])
    B = np.array([[10,20,30],
                  [5,6,7]])

    cartesian_product_2D( A, B, axis=1 )
    >> np.array([[ 1*10, 1*20, 1*30, 2*10, 2*20, 2*30 ]
                 [ 3*5,  3*6,  3*7,  4*5,  4*6,  4*7  ]])
    :param arrays:  list of (N, D) numpy arrays
                    Each array must have the same N in the axis that is not being multiplied across
                    I.e. if axis = 1, each array in the list must be of dimension (N,D1), (N,D2)... (N,Dn)
    :param axis:    int
                    The axis across which to perform the product
                    axis = 0 means we want combinations down the rows
                    axis = 1 means we want combinatinos across the columns
    :return:
    """
    A_shape = tf.shape(A)  # N = A.shape[1].value
    if axis == 0:  # For combinations down the rows
        A_ = tf.expand_dims(A, axis=0)
        B_ = tf.expand_dims(B, axis=1)
        out = tf.reshape(tf.multiply(A_, B_), [-1, A_shape[1]])
    elif axis == 1:  # For combinations across the columns
        A_ = tf.expand_dims(A, axis=1)
        B_ = tf.expand_dims(B, axis=2)
        out = tf.reshape(tf.multiply(A_, B_), [A_shape[0], -1])
    else:
        raise ValueError("axis must be 0 or 1")
    return out


def neg_log_marginal_likelihood_explicit(N,
                                         M,
                                         PHI,
                                         Y_trn,
                                         alpha,
                                         beta,
                                         S,
                                         Sinv):
    """
    The negative log marginal likelihood for bayesian linear regression
    :param Y_trn:
    :param PHI: NxM ndarray
                defined as phi(X_trn)
    :param alpha:
    :param beta:
    :param S:
    :param Sinv:
    :param M:
    :return:

    ref:
    Lazaro-Gredilla et al, 2010, JMLR & Bishop 2006
     "Sparse spectrum Gaussian process regression"
    """

    m = M // 2
    sign_det, log_abs_det = tf.linalg.slogdet(Sinv)
    calc1 = tf.matmul(tf.matmul(Y_trn, PHI, transpose_a=True), S)
    calc2 = tf.matmul(calc1, PHI, transpose_b=True)
    calc3 = tf.matmul(calc2, Y_trn)
    yy = tf.matmul(Y_trn, Y_trn, transpose_a=True)
    p1 = (yy - calc3 * beta) * (beta / 2.0)
    p2 = m * tf.log(beta)
    p3 = (1 / 2.0) * sign_det * log_abs_det
    p4 = m * tf.log(alpha / beta)
    p5 = (N / 2.0) * tf.log(2 * np.pi / beta)

    nlml = -1 * (-p1 + p2 - p3 + p4 - p5)
    return nlml


def neg_log_marginal_likelihood_loss(Y_pred,
                                     mu,
                                     N,
                                     M,
                                     Y_trn,
                                     alpha,
                                     beta,
                                     S):
    """
    The negative log marginal likelihood for bayesian linear regression
    :param Y_pred:
    :param N:
    :param M:
    :param Y_trn:
    :param alpha:
    :param beta:
    :param S:

    :return:

    ref:
    Lazaro-Gredilla et al, 2010, JMLR & Bishop 2006
     "Sparse spectrum Gaussian process regression"
    """
    sign_det, log_abs_det = tf.linalg.slogdet(S)
    # because we're doing log(1/|S|) instead of log(|Sinv|)
    sign_det *= -1

    Y_hat = tf.reduce_mean(Y_pred, 1, keepdims=True)
    p1 = (M / 2) * tf.log(alpha)
    p2 = (N / 2) * tf.log(beta)
    E_mn1 = (beta / 2.0) * tf.square(tf.norm(Y_trn - Y_hat, keepdims=False))
    E_mn2 = (alpha / 2.0) * tf.matmul(mu, mu, transpose_a=True)
    p4 = (1 / 2.0) * sign_det * log_abs_det
    # p5 = (N / 2.0) * tf.log(2 * np.pi)
    nlml = -1 * (p1 + p2 - E_mn1 - E_mn2 - p4)  # - p5)
    return nlml


def logdet_correction(A_logdet,
                      S,
                      U,
                      V,
                      tfdt=tf.float64):
    """
    NOTE: V here is already V.transpose()d
    :return:
    """
    rank = tf.shape(V)[0]
    VA = tf.matmul(V, S)
    sign_det, log_det = tf.linalg.slogdet(tf.eye(rank, dtype=tfdt) +
                                          tf.matmul(VA, U))
    return tf.assign_add(A_logdet, sign_det * log_det)


def smw_nlml(Y_pred,
              mu,
              N,
              M,
              Y_trn,
              alpha,
              beta,
              logdet,
              tfdt=tf.float64):
    """
    Sherman-Morrison-Woodbury NLML
    NOTE: This is the loss as defined in Bishop
    :param Y_pred:
    :param mu:
    :param N:
    :param M:
    :param Y_trn:
    :param alpha:
    :param beta:
    :param logdet:
    :param tfdt:
    :return:
    """
    Y_hat = tf.reduce_mean(Y_pred, 1, keepdims=True)
    p1 = (M / 2) * tf.log(alpha)
    p2 = (N / 2) * tf.log(beta)
    E_mn1 = (beta / 2.0) * tf.square(tf.norm(Y_trn - Y_hat, keepdims=False))
    E_mn2 = (alpha / 2.0) * tf.matmul(mu, mu, transpose_a=True)
    p4 = (1 / 2.0) * logdet
    p5 = (N / 2.0) * tf.log(2 * tf.constant(np.pi,
                                            dtype=tfdt))
    nlml = -1 * (p1 + p2 - E_mn1 - E_mn2 - p4 - p5)
    return nlml


def smw_inv_precomp(A_inv, VS, R_lower, tfdt=tf.float32):
    """
    Faster iterative SMW using cholesky solve for inversion
    :param A_inv:
    :param VS:
    :param R_lower:
    :param tfdt:
    :return:
    """
    solv1 = tf.matrix_solve_ls(R_lower,
                               tf.eye(tf.shape(R_lower)[0], dtype=tfdt))
    I_plus_VSU_inv = tf.matrix_solve_ls(tf.transpose(R_lower), solv1)
    SU_I_plus_VSU = tf.matmul(VS, I_plus_VSU_inv, transpose_a=True)
    return tf.assign_sub(A_inv, tf.matmul(SU_I_plus_VSU, VS))


def logdet_correction_precomp(A_logdet, R_lower):
    """
    Fast iterative logdet using cholesky decomposed A
    :param A_logdet:
    :param R_lower:
    :return:
    """
    logdet_chol = (1.0 / 2.0) * tf.reduce_sum(
        tf.log(tf.square(tf.diag_part(R_lower))), keepdims=False)
    return tf.assign_add(A_logdet, logdet_chol)
