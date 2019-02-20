import numpy as np
from scipy import interpolate
from scipy.interpolate import CubicSpline, PchipInterpolator


def interp_with_asymptotes(xy, x):
    iwa = InterpWithAsymptotes(xy[:, 0], xy[:, 1])
    return iwa(x)


class InterpWithAsymptotes:
    """
    Wraps a scipy interpolator to add extrapolation with asymptotes at x=-1
    and x = 1.

    usage
    -----
    # Points
    pts = np.array([
        [0.12, 0.2, 0.5, 0.7, 0.9],
        [-0.5, 1.0, 0.0, 0.12, 0.3]]).T

    # Interpolator
    interp = InterpWithAsymptotes(pts[:, 0], pts[:, 1])


    plt.figure(figsize=(14, 8))
    x = np.linspace(0, 1, 1000)
    plt.plot(x, interp(x), 'b')
    plt.plot(pts[:, 0], pts[:, 1], 'ro')
    plt.axvline(x=0, c='k')
    plt.axvline(x=1, c='k')
    plt.ylim(-3, 3)
    plt.xlim(-0.1, 1.1)
    plt.show()

    """

    def __init__(self, x, y, interpolator=interpolate.PchipInterpolator,
                 params=None):
        """
        :param x: x coordinates of points to interpolate
        :param y: Y coordinates of points to interpolate
        :param interpolator: scipy interpolator class
        """
        if params is None:
            self.interp = interpolator(x=x, y=y, extrapolate=True)
        else:
            self.interp = interpolator(x=x, y=y, extrapolate=True,
                                       params=params)

        # First and second derivatives for first and last point
        ext_pt = np.array([x[0], x[-1]])
        ext_pt_y = np.array([y[0], y[-1]])
        dervs = self.interp(ext_pt, 1)
        small_der = (dervs < 0.01)
        dervs = dervs * ~small_der + 1.0 * small_der
        # dervs2 = interp(ext_pt, 2)

        inner = abs(.5 * np.sign(ext_pt - .5) + .5 - ext_pt)
        self.aa = np.sign(ext_pt - .5) * dervs * (inner ** 2)
        self.bb = ext_pt_y - self.aa / inner
        self.firstPt, self.lastPt = ext_pt[0], ext_pt[1]

    def __call__(self, x):
        is_left = x < self.firstPt
        is_right = x > self.lastPt
        is_between = (~is_left) * (~is_right)

        v_left = is_left * (self.bb[0] + self.aa[0] / x)
        v_right = is_right * (self.bb[1] + self.aa[1] / (1 - x))
        v_between = is_between * self.interp(x)
        return v_left + v_between + v_right


class HandleCubicSpline(PchipInterpolator):
    """
    Cubic spline interpolation with handles on first and last point. Handles
    constrain the first derivatives (x and y).

    usage
    -----
    x = np.array([0.04, 0.5, 0.96])
    y = np.array([-1, 0.0, 1])
    params = np.array([0.1, 2.0, 0.1, 2.0])

    interp = HandleCubicSpline(x, y)

    plt.figure()
    plt.plot(x, y, 'ro')
    lin = np.linspace(0, 1, 100)
    spl = interp(lin)
    plt.plot(lin, spl)
    plt.show()

    """

    def __init__(self, x, y, params, pchip_samples=30, **kwargs):
        """
        :param x: x coordinates of points to interpolate
        :param y: Y coordinates of points to interpolate
        :param params: vector of four derivatives for start and end point:
        (d0x, d0y, dNx, dNy)
        :param pchip_samples: number of samples for Pchip interpolation
        """
        t = np.linspace(0, 1, len(x))
        y_coord = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])

        # Fit parametric 2D cubic spline
        sp = CubicSpline(t, y_coord, bc_type=(
            (1, (params[0], params[1])),
            (1, (params[2], params[3]))))

        # Fit a pchipInterpolator to cubic spline
        samples_t = np.linspace(0, 1, pchip_samples)
        samples = sp(samples_t)
        super().__init__(x=samples[:, 0], y=samples[:, 1], extrapolate=True)


class PieceWiseLinear(interpolate.interp1d):
    def __init__(self, x, y, **kwargs):
        """
        :param x: x coordinates of points to interpolate
        :param y: Y coordinates of points to interpolate
        """
        self.xmin, self.xmax = np.min(x), np.max(x)
        self.eps = 0.001 * (self.xmax - self.xmin)
        super().__init__(x, y)
        self.ymin = y[np.argmin(x)]
        self.ymax = y[np.argmax(x)]
        self.dermin = (super().__call__(self.xmin + 2 * self.eps) -
                       super().__call__(self.xmin)) / self.eps
        self.dermax = (super().__call__(self.xmax) -
                       super().__call__(self.xmax - 2 * self.eps)) / self.eps

    def __call__(self, x, nu=0):
        if nu == 0:
            return self.__call_helper(x)
        elif nu == 1:
            d = self.__call_helper(x + self.eps) - \
                self.__call_helper(x - self.eps)
            return .5 * d / self.eps
        else:
            raise NotImplementedError("Only handles first derivative.")

    def __call_helper(self, x):
        """
        This function handles extrapolation because scipy doesnt.
        The scipy call is only done for data between xmin and xmax.
        """
        mask_lower = (x < self.xmin)
        mask_higher = (x > self.xmax)
        mask_else = (~mask_lower) * (~mask_higher)
        d_lower = self.xmin - x
        d_higher = x - self.xmax

        extra_higher = self.ymax + self.dermax * d_higher
        extra_lower = self.ymin - self.dermin * d_lower
        id_intra = np.where(mask_else)
        intra_vals = super().__call__(x[id_intra])
        intra = np.zeros((extra_lower.shape))
        intra[id_intra] = intra_vals

        return mask_lower * extra_lower + mask_higher * extra_higher \
               + mask_else * intra
