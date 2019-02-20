import numpy as np


def log_space_mirrored(start=0.1, stop=1.0, centre=0.5, num=101, endpoint=True,
                       base=10.0, dtype=np.float64):
    """

                                  [start]       [centre]           [stop]
    I.e. the spacing looks like   [x]x x   x      [x]      x   x  x[x]
    USAGE: a = logspace_centre_mirrored(start=0.1, stop=1.0, centre=0.5,
           num=101)

    :param start:
    :param stop:
    :param centre:
    :param num:          integer (odd)
    :param endpoint:
    :param base:
    :param dtype:
    :return:
    """
    num_left = 1 + num // 2
    # num_right = num // 2
    sequence = np.empty(shape=num, dtype=dtype)
    range_left = centre - start
    range_right = stop - centre
    unit_sequence_left = (np.logspace(start=0, stop=1, num=num_left,
                                      endpoint=endpoint, base=base,
                                      dtype=dtype) - 1) / 9.0
    # Ignore the middle; re-use the midpoint of left sequence
    unit_sequence_right = (1 - unit_sequence_left[:-1])[::-1]

    left_sequence = start + unit_sequence_left * range_left
    right_sequence = centre + unit_sequence_right * range_right
    sequence[0:num_left] = left_sequence
    sequence[num_left:] = right_sequence
    return sequence
