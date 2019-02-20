import numpy as np

DEFAULT_NOISE = 0.02


def noisy(v, var_noise):
    return v + np.random.normal(0, var_noise, v.shape)


"""
Interpolation functions
"""


def heaviside(x, var_noise=DEFAULT_NOISE):
    v = -.5 + .5 * (x > .5)
    return noisy(v, var_noise)


def steps(x, var_noise=DEFAULT_NOISE):
    v = .2 * (x > 0) + .2 * (x > .3) + .2 * (x > .6) + .2 * (x > .9)
    return noisy(v, var_noise)


def quadratic_cos(x, var_noise=DEFAULT_NOISE):
    v = .3 * np.cos(30 * x) + .3 * (2 * (x - 0.5)) ** 2 - 0.1
    return noisy(v, var_noise)


"""
Periodic functions
"""


def cosine(x, var_noise=DEFAULT_NOISE):
    v = .3 * np.cos(30 * x)
    return noisy(v, var_noise)


def pattern(x, var_noise=DEFAULT_NOISE):
    v = .3 * np.cos(10 * x) - .4 * np.clip(np.cos(20 * x), 0, None)
    v_clip = np.clip(v, -0.2, None)
    return noisy(v_clip, var_noise)


def harmonics(x, var_noise=DEFAULT_NOISE):
    v = .3 * np.cos(20 * x) + .2 * np.cos(40 * x + 1) + .1 * np.cos(60 * x + 2)
    return noisy(v, var_noise)
