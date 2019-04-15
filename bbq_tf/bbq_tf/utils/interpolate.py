from .math import diff
import tensorflow as tf
from tensorflow import SparseTensor as STensor, gather


def pchiptx_asymp(x, y, x_query, tfdt=tf.float64):
    """

    :param x:   (N,)
                All the interpolation "training" x values
    :param y:   (N,)
                All the interpolation "training" y values

    :param x_query:
    :return:
    """
    # All parts of x_query > x.max():
    #       Extrapolate these points to the right asymptote
    # all parts of x_query < x.min():
    #       Extrapolate these points to the left asymptote
    # All the parts x.min() <= x_query <= x.max():
    #       Extrapolate these points with pchiptx
    _x_query = tf.reshape(x_query, (-1,))

    n = _x_query.shape[0].value

    px_right = x[-1]
    px_left = x[0]
    py_right = y[-1]
    py_left = y[0]

    left_mask = tf.less(_x_query, px_left)
    middle_mask = tf.logical_and(tf.greater_equal(_x_query, px_left),
                                 tf.less_equal(_x_query, px_right))
    right_mask = tf.greater(_x_query, px_right)

    right_query = tf.boolean_mask(_x_query, right_mask)
    middle_query = tf.boolean_mask(_x_query, middle_mask)
    left_query = tf.boolean_mask(_x_query, left_mask)

    middle_interp, d = pchiptx(x, y, middle_query, tfdt=tfdt)

    deriv_right = d[-1]
    deriv_left = d[0]

    a_right = deriv_right * (px_right - 1) ** 2
    a_left = deriv_left * (px_left) ** 2
    right_interp = - a_right / (right_query - 1.0) + a_right / (
            px_right - 1) + py_right
    left_interp = - a_left / (left_query) + a_left / (px_left) + py_left

    reconstruct_coords = tf.concat(
        [tf.where(left_mask), tf.where(middle_mask), tf.where(right_mask)],
        axis=0)
    reconstruct_zeropad = tf.zeros(tf.shape(reconstruct_coords), dtype=tf.int64)
    reconstruct_idxs = tf.concat([reconstruct_zeropad, reconstruct_coords],
                                 axis=1)

    scatter_shape = tf.constant([1, n])
    reconstruct_values = tf.concat([left_interp, middle_interp, right_interp],
                                   axis=0)

    combined = tf.scatter_nd(indices=tf.cast(reconstruct_idxs, tf.int32),
                             updates=reconstruct_values,
                             shape=scatter_shape)

    return tf.reshape(combined, tf.shape(x_query))


def pchiptx_full(x, y, x_query):
    """
    The pchip interpolation
    Ported from matlab's pchiptx
    :param x:       Tensor (n,)
                    The x positions of the points for interpolation
    :param y:       Tensor (n,)
                    The y positions of the points for interpolation
    :param x_query: Tensor (M,)
                    The query points we want to interpolate at
    :return:

    Example usage
    -------------
    if __name__ == "__main__":
        tfdt = tf.float32
        x = tf.constant(np.array([1., 2., 4., 8., 9.]), dtype=tfdt)
        y = tf.constant(np.array([0.1, 0.9, 0.9, 3.0, 12.0]), dtype=tfdt)
        x_query = tf.constant(np.linspace(1.0, 9.0, 7), dtype=tfdt)
        sess = tf.Session()
        with sess.as_default():  # or `with sess:` to close on exit
            assert sess is tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        query_interpolated = sess.run(pchiptx(x, y, x_query))
        print("Interpolated y_query: {}".format(query_interpolated))
    """
    #  First derivatives
    # Flatten to allow easy application to matrices
    _x_query = tf.reshape(x_query, (-1,))
    h = diff(a=x, axis=0)
    delta = diff(a=y, axis=0) / h
    d = pchipslopes(h, delta)

    #  Piecewise polynomial coefficients
    N = x.shape[0].value
    c = (3 * delta - 2 * d[0:N - 1] - d[1:N]) / h
    b = (d[0:N - 1] - 2 * delta + d[1:N]) / (h ** 2)

    #  Find subinterval indices k so that x(k) <= u < x(k+1)
    j_range = tf.reshape(tf.range(1, N - 1, dtype=tf.int32), (-1, 1))
    K = tf.reduce_max(
        tf.cast(tf.less_equal(gather(x, j_range), _x_query),
                tf.int32) * j_range,
        axis=0, keepdims=False)

    #  Evaluate interpolant
    s = _x_query - gather(x, K)
    v = gather(y, K) + \
        s * (gather(d, K) +
             s * (gather(c, K) +
                  s * gather(b, K)))
    # Reshape to original query shape
    return tf.reshape(v, tf.shape(x_query))


def pchiptx(x, y, x_query, tfdt=tf.float64):
    """
    The pchip interpolation
    Ported from matlab's pchiptx
    :param x:       Tensor (n,)
                    The x positions of the points for interpolation
    :param y:       Tensor (n,)
                    The y positions of the points for interpolation
    :param x_query: Tensor (M,)
                    The query points we want to interpolate at
    :return:

    Example usage
    -------------
    if __name__ == "__main__":
        tfdt = tf.float32
        x = tf.constant(np.array([1., 2., 4., 8., 9.]), dtype=tfdt)
        y = tf.constant(np.array([0.1, 0.9, 0.9, 3.0, 12.0]), dtype=tfdt)
        x_query = tf.constant(np.linspace(1.0, 9.0, 7), dtype=tfdt)
        sess = tf.Session()
        with sess.as_default():  # or `with sess:` to close on exit
            assert sess is tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        query_interpolated = sess.run(pchiptx(x, y, x_query))
        print("Interpolated y_query: {}".format(query_interpolated))
    """
    #  First derivatives
    # Flatten to allow easy application to matrices
    h = diff(a=x, axis=0)
    delta = diff(a=y, axis=0) / h
    d = pchipslopes(h, delta, tfdt=tfdt)

    #  Piecewise polynomial coefficients
    N = x.shape[0].value
    c = (3 * delta - 2 * d[0:N - 1] - d[1:N]) / h
    b = (d[0:N - 1] - 2 * delta + d[1:N]) / (h ** 2)

    # #  Find subinterval indices k so that x(k) <= u < x(k+1)
    j_range = tf.reshape(tf.range(1, N - 1, dtype=tf.int32), (-1, 1))
    K = tf.reduce_max(
        tf.cast(tf.less_equal(gather(x, j_range), x_query),
                tf.int32) * j_range,
        axis=0, keepdims=False)

    # #  Evaluate interpolant
    s = x_query - gather(x, K)
    v = gather(y, K) + \
        s * (gather(d, K) +
             s * (gather(c, K) +
                  s * gather(b, K)))
    # Reshape to original query shape
    return v, d


def pchipslopes(h, delta, tfdt=tf.float64):
    """

    :param h:
    :param delta:
    :return:
    """
    n = h.shape[0].value
    val_to_check = tf.sign(delta[0:n - 1]) * tf.sign(delta[1:n])
    k = tf.where(tf.greater(val_to_check, 0.0))
    k_flat = k[:, 0] + 1  # k_flat = k[0] + 1
    k_idx = tf.concat([tf.zeros(shape=tf.shape(k), dtype=tf.int64), k], axis=1)
    w1 = 2 * gather(h, k_flat) + gather(h, k_flat - 1)
    w2 = gather(h, k_flat) + 2 * gather(h, k_flat - 1)
    d_middle_values = (w1 + w2) / (
            w1 / gather(delta, k_flat - 1) + w2 / gather(delta, k_flat))
    scatter_shape = tf.constant([1, n - 1])
    d_middle = tf.scatter_nd(indices=tf.cast(k_idx, tf.int32),
                             updates=d_middle_values,
                             shape=scatter_shape)
    #  Slopes at endpoints
    d_0 = pchipend(h[0], h[1], delta[0], delta[1], tfdt=tfdt)
    d_n = pchipend(h[n - 1], h[n - 2], delta[n - 1], delta[n - 2], tfdt=tfdt)

    d = tf.concat([tf.expand_dims(d_0, 0), d_middle[0], tf.expand_dims(d_n, 0)],
                  axis=0)
    return d


def pchipend_cond1(tfdt=tf.float64):
    return tf.cast(0.0, tfdt)


def pchipend_cond2(del1, tfdt=tf.float64):
    return tf.cast(3.0 * del1, tfdt)


def pchipend_default(h1, h2, del1, del2, tfdt=tf.float64):
    return tf.cast(((2.0 * h1 + h2) * del1 - h1 * del2) / (h1 + h2), tfdt)


def pchipend(h1, h2, del1, del2, tfdt=tf.float32):
    #  Noncentered, shape-preserving, three-point formula.
    d = tf.cast(((2.0 * h1 + h2) * del1 - h1 * del2) / (h1 + h2), tfdt)
    r = tf.case({tf.not_equal(tf.sign(d),
                              tf.sign(del1)): lambda: pchipend_cond1(tfdt=tfdt),
                 tf.logical_and(tf.not_equal(tf.sign(del1), tf.sign(del2)),
                                tf.greater(tf.abs(d),
                                           tf.abs(3.0 * del1))
                                ): lambda: pchipend_cond2(del1, tfdt=tfdt)},
                default=lambda: pchipend_default(h1, h2, del1, del2, tfdt=tfdt),
                exclusive=False,
                )

    return r
