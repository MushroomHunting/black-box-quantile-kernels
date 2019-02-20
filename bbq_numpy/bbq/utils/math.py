import numpy as np


def smw_inv_correction(a_inv, u, v):
    """
    Sherman-Morrison-Woodbury update
    For rank k updates to the inverse matrix

    IMPORTANT: This is the correction factor which one must subtract from A_inv
    Usage:   subtract this value from current A_inv

    ref:     http://mathworld.wolfram.com/WoodburyFormula.html
             https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    :param a_inv:   n x n
    :param u:      n x k
    :param v:      k x n
    :return:
    """
    rank = u.shape[1]
    su = np.dot(a_inv, u)
    vs = np.dot(v, a_inv)
    i_plus_vsu_inv = np.linalg.pinv(np.identity(rank) + np.dot(vs, u))
    su_i_plus_vsu = np.dot(su, i_plus_vsu_inv)
    return np.dot(su_i_plus_vsu, vs)


def qf_to_pdf(x_qf, y_qf, regenerate_cdf=False):
    """
    :param x_qf: \in [0,1]
    :param y_qf: \in (-\infty, +\infty)
    :param regenerate_cdf: the cdf from the reconstructed pdf - a sanity check
    :return: (x-axis labels, y-axis labels) to plot
    """
    pdf = np.diff(x_qf)
    x_diff = (y_qf[1:] - y_qf[:-1])
    # in a case there are invalid quantiles e.g. cdf(i) = cdf(i+1)
    valid_qf_points = np.logical_not(np.isclose(x_diff, 0))
    pdf = np.divide(pdf[valid_qf_points], x_diff[valid_qf_points])
    x_plot, y_plot = y_qf[:-1][valid_qf_points], pdf

    if regenerate_cdf is True:
        cdf = np.zeros(len(y_plot))
        for i in range(len(y_plot)):
            # do not provide x-axis if the x_resolution is very high because
            # trapz can't handle it
            cdf[i] = np.trapz(y_plot[:i], x=x_plot[:i])
            print('y=', x_plot[i], y_plot[i], cdf[i])

        max_cdf = np.trapz(y_plot[:-1], x=x_plot[:-1])
        print("max of cdf=", max_cdf)
        # TODO: take out, once inf,nan, etc. are filtered
        # check if the max value of the cdf is close to 1
        if np.isclose(max_cdf, 1, rtol=0.05) is False:
            print("BBQ Err: Check if qf_to_pdf is correct or"
                  "if the quantile is valid!")

    return x_plot, y_plot


def neg_log_marginal_likelihood(phi, y_trn, alpha, beta, s, s_inv):
    """
    The negative log marginal likelihood for bayesian linear regression
    Important: This needs to be minimized!
    Alternatively, the LML must be maximized
    :param y_trn:
    :param phi: NxM ndarray
                defined as phi(X_trn)
    :param alpha:
    :param beta:
    :param s:
    :param s_inv:
    :return:

    ref:
    Lazaro-Gredilla et al, 2010, JMLR & Bishop 2006
     "Sparse spectrum Gaussian process regression"
    """
    n, m = phi.shape
    half_m = m // 2
    s_log_det = np.linalg.slogdet(s_inv)
    calc1 = np.dot(np.dot(y_trn.T, phi), s)
    calc2 = np.dot(calc1, phi.T)
    calc3 = np.dot(calc2, y_trn)
    yy = np.dot(y_trn.T, y_trn)
    p1 = (yy - calc3 * beta) * (beta / 2.0)
    p2 = half_m * np.log(beta)
    p3 = (1 / 2.0) * s_log_det[0] * s_log_det[1]
    p4 = half_m * np.log(alpha / beta)
    p5 = (n / 2.0) * np.log(2 * np.pi / beta)

    nlml = -1 * (-p1 + p2 - p3 + p4 - p5)
    return nlml


def cartesian_product(arrays):
    """
    wraps our math.cartesian_product to work on 2D matrices where we want to
    apply the cartesian product
    across a particular axis, and not each value

    Note
    ----
    numpexr modification from the so post
    This is probably the most efficient full parallelized numpy-based method
    https://stackoverflow.com/questions/44323478/efficient-axis-wise-cartesian-product-of-multiple-2d-matrices-with-numpy-or-tens

    Example
    -------
    A = np.array([[1,2],
                  [3,4]])
    B = np.array([[10,20],
                  [5,6]])
    C = np.array([[50, 0],
                  [60, 8]])
    cartesian_product_2D( [A,B,C], axis=1 )
    >> np.array([[ 1*10*50, 1*10*0, 1*20*50, 1*20*0, 2*10*50, 2*10*0, 2*20*50,
                  2*20*0]
                 [ 3*5*60,  3*5*8,  3*6*60,  3*6*8,  4*5*60,  4*5*8,  4*6*60,
                 4*6*8]])
    :param arrays:  list of (N, D) numpy arrays
                    Each array must have the same N in the axis that is not
                    being multiplied across
                    I.e. if axis = 1, each array in the list must be of
                    dimension (N,D1), (N,D2)... (N,Dn)
    :return:
    """
    if len(arrays) == 1:
        return arrays[0]
    n = arrays[0].shape[0]
    a = arrays[1][:, None]  # temp
    b = arrays[0][:, :, None]  # temp
    out = (a * b).reshape(n, -1)
    # out = ne.evaluate("a*b").reshape(n, -1)
    for i in arrays[2:]:
        a = i[:, None]  # temp
        b = out[:, :, None]  # temp
        out = (a * b).reshape(n, -1)
        # out = ne.evaluate("a*b").reshape(n, -1)
    return out
