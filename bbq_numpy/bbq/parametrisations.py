import numpy as np
from bbq.interpolate import HandleCubicSpline, PieceWiseLinear
from scipy.interpolate import PchipInterpolator
from scipy.stats import dweibull


class AbsParametrisation:
    """
    Abstract parametrisation class helper. All parametrisations should
    implement this class.
    """

    def __init__(self):
        self.interpolator = None
        self.paramLow = None
        self.paramHigh = None
        self.logScaleParams = None

    def gen_params(self, x):
        raise NotImplementedError()

    def scale_params(self, params):
        log_params = 10 ** params
        return log_params * self.logScaleParams + params * (
            ~self.logScaleParams)


class InterpSingleSpline(AbsParametrisation):
    """
    The quantile function is a single cubic spline. This is a symetric
    parametrisation. Parameters govern y coordinate of end points, and the
    slope derivative at end points.
    """

    def __init__(self):
        AbsParametrisation.__init__(self)
        self.eps = 0.03
        self.interpolator = HandleCubicSpline
        self.paramLow, self.paramHigh = np.array([0.1, 0]), np.array([2.3, 8])
        self.logScaleParams = np.array([True, False])

    def gen_params(self, x):
        ptx = np.array([self.eps, 0.5, 1 - self.eps])
        pty = np.array([-x[0], 0.0, x[0]])
        der = x[0] * x[1]
        p = np.array([0, der, 0, der])
        return ptx, pty, p


class InterpPieceWise4points(AbsParametrisation):
    """
    The quantile function is piece-wise linear, with 4 points. Parameters
    represent the y coordinate of end points, and the x (distance to center)
    and y coordinate of center points.
    """

    def __init__(self):
        AbsParametrisation.__init__(self)
        self.eps = 0.03
        self.interpolator = PieceWiseLinear
        self.paramLow = np.array([0.01, 1.0, 0.0])
        self.paramHigh = np.array([0.45, 2.3, 1.0])
        self.logScaleParams = np.array([False, True, False])

    def gen_params(self, x):
        ptx = np.array([self.eps, 0.5 - x[0], 0.5 + x[0], 1 - self.eps])
        pty = np.array([-x[1], -x[1] * x[2], x[1] * x[2], x[1]])
        return ptx, pty, None


class InterpIncrementY6pts(AbsParametrisation):
    """
    The quantile is interpolated with pcphi, using 6 points. Parameters
    represent the y coordinates increments, and the last one is a stretch
    along the y axis.
    """

    def __init__(self):
        AbsParametrisation.__init__(self)
        self.eps = 0.01
        self.interpolator = PchipInterpolator
        self.paramLow = np.array([0, 0, 0, 0.1])
        self.paramHigh = np.array([1, 1, 1, 2.3])
        self.logScaleParams = np.array([False, False, False, True])

    def gen_params(self, x):
        ptx = np.array([self.eps, 2 * self.eps, 0.5 - self.eps,
                        0.5 + self.eps, 1 - 2 * self.eps, 1 - self.eps])
        pty = np.array([0,
                        x[0],
                        x[0] + x[1],
                        x[0] + x[1] + x[2],
                        x[0] + x[1] + x[2] + x[1],
                        x[0] + x[1] + x[2] + x[1] + x[0]])
        pty -= (pty[2] + pty[3]) / 2.0
        pty *= x[3]  # / (pty[-1] - pty[0])
        return ptx, pty, None


class PeriodicSimple(AbsParametrisation):
    """
    The quantile function is piece-wise linear. Parameters represent the
    distance between mid points and 0.5, and the y coordinate of end points.
    """

    def __init__(self, use_pchip_interpolation=False):
        AbsParametrisation.__init__(self)
        self.eps = 0.03
        if use_pchip_interpolation:
            self.interpolator = PchipInterpolator
        else:
            self.interpolator = PieceWiseLinear
        self.paramLow = np.array([1.2, 0])
        self.paramHigh = np.array([2.0, 0.1])
        self.logScaleParams = np.array([True, False])

    def gen_params(self, x):
        ptx = np.array([self.eps, 0.5 - x[1], 0.5 + x[1], 1 - self.eps])
        pty = np.array([-x[0], -x[0], x[0], x[0]])
        return ptx, pty, None


class InterpWeibull(AbsParametrisation):
    """
    The quantile function is the one from the weibull distribution.
    Parameters are those of the weibull distribution.
    """

    class WeibullInterpolator:
        def __init__(self, args, *kwargs):
            self.params = kwargs['params']

        def __call__(self, x, nu=0):
            return dweibull.ppf(x, c=self.params[0], loc=self.params[1],
                                scale=self.params[2])

    def __init__(self):
        AbsParametrisation.__init__(self)
        self.paramLow = np.array([0, 0, 0])
        self.paramHigh = np.array([1, 1, 1])
        self.logScaleParams = np.array([False, False, False])
        self.interpolator = self.WeibullInterpolator

    def gen_params(self, x):
        # TODO
        return None, None, x


class BoundedQPoints(AbsParametrisation):
    def __init__(self, n, p1=(0.025, -10), p2=(0.975, 10)):
        AbsParametrisation.__init__(self)
        self.N = n
        eps = 0.01
        # self.interpolator = PieceWiseLinear
        self.interpolator = PchipInterpolator
        self.paramLow = np.array([eps] * (self.N - 1) * 2)
        self.paramHigh = np.array([1.0] * (self.N - 1) * 2)
        self.logScaleParams = np.array([False] * (self.N - 1) * 2)

        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.xy_ranges = np.array([[p2[0] - p1[0], p2[1] - p1[1]]])
        # 2D offset coordinates
        self.offsets = np.random.uniform(0, 1, size=(n - 1, 2))
        self.rescale_factor = np.zeros(shape=(1, 2))
        self.points = np.zeros(shape=(n, 2))
        self.__refresh_points()  # Initialise the points

    def __normalise_points(self):
        self.rescale_factor = np.sum(self.offsets, axis=0, keepdims=True)
        self.points = self.p1 + self.points * (
                self.xy_ranges / self.rescale_factor)

    def __refresh_points(self):
        for i in range(1, self.N):
            self.points[i] += self.points[i - 1] + self.offsets[i - 1]
        self.__normalise_points()

    def gen_params(self, x):
        self.points = np.zeros(shape=(self.N, 2))
        self.offsets[:, 0] = x[0:self.N - 1]
        self.offsets[:, 1] = x[self.N - 1:]
        self.__refresh_points()
        return self.points[:, 0], self.points[:, 1], None


class StairCase(AbsParametrisation):
    def __init__(self, n, p1=(0.025, -10), p2=(0.975, 10)):
        AbsParametrisation.__init__(self)
        self.x_min, self.x_max = p1[0], p2[0]
        self.y_min, self.y_max = p1[1], p2[1]
        self.n = n
        self.interpolator = PieceWiseLinear
        self.paramLow = np.array([0] * 2 * n)
        self.paramHigh = np.array([1] * 2 * n)
        self.logScaleParams = np.array([False] * 2 * n)

    @staticmethod
    def __normalise_points(x, a=0, b=1):
        return np.cumsum(x) * (b - a) / np.sum(x) + a

    def gen_params(self, x):
        midx = StairCase.__normalise_points(x[:self.n + 1], self.x_min,
                                            self.x_max)[:-1]
        pert_x = np.repeat(midx, 2)
        pert_x[1::2] += 0.001
        ptx = np.array([self.x_min] + list(pert_x) + [self.x_max])
        midy = StairCase.__normalise_points(x[self.n:], self.y_min, self.y_max)
        pty = np.array([self.y_min] * 2 + list(np.repeat(midy, 2)))
        return ptx, pty, None
