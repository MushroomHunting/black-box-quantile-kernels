import numpy as np


class Normaliser:
    """
    Normalise/unnormalise data to [0,1] or [-1,1].
    """

    def __init__(self, low, high, zero_one_interval=True):
        """
        :param low: List of lower-bounds for each dimension
        :param high: List of upper-bounds for each dimension
        :param zero_one_interval: whether normalised interval should be [0,1]
        (default) or [-1,1]
        """
        assert (len(low) == len(high) and
                "Upper and lower bounds much be same dimension.")
        assert (np.isfinite(np.sum(low)) and
                "Lower bound elements must be numbers.")
        assert (np.isfinite(np.sum(high)) and
                "Upper bound elements must be numbers.")

        space_range = np.array(high) - np.array(low)

        if np.sum(space_range > 100) > 0:
            print("Warning: normalising over large space.")

        self.factor = (1.0 if zero_one_interval else 2.0) * space_range
        self.invFactor = (1.0 if zero_one_interval else 2.0) / space_range
        self.offset = -np.array(low)
        self.finalOffset = 0.0 if zero_one_interval else -1.0
        self.bounds_norm = (space_range * 0 - (0 if zero_one_interval else 1),
                            space_range * 0 + 1)
        self.bounds_orig = (np.array(low), np.array(high))

    def normalise(self, x):
        """
        Normalise x.
        :param x: list with 1 element, or N*D numpy matrix with N elements
        :return: numpy matrix with shape of input
        """
        _x = np.array(x)
        if len(_x.shape) == 1:
            assert (_x.shape == self.offset.shape and
                    "Data must be same dimension as lower/upper bounds")
        else:
            assert (_x.shape[1] == self.offset.shape[0] and
                    "Data must be same dimension as lower/upper bounds")

        return (_x + self.offset) * self.invFactor + self.finalOffset

    def unnormalise(self, x):
        """
        Unnormalise x.
        :param x: list with 1 element, or N*D numpy matrix with N elements
        :return: numpy matrix with shape of input
        """
        _x = np.array(x)
        if len(_x.shape) == 1:
            assert (_x.shape == self.offset.shape and
                    "Data must be same dimension as lower/upper bounds")
        else:
            assert (_x.shape[1] == self.offset.shape[0] and
                    "Data must be same dimension as lower/upper bounds")

        return (_x - self.finalOffset) * self.factor - self.offset

    def bounds_normalised(self):
        return self.bounds_norm

    def bounds_original(self):
        return self.bounds_orig
