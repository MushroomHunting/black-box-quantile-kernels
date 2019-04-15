class QMC_KWARG(object):
    PERM = "PERM"  # Permutation (for generalised halton)


class QMC_SCRAMBLING(object):
    GENERALISED = "GENERALISED"
    OWEN17 = "OWEN17"

class QMC_SEQUENCE(object):
    HALTON = "HALTON"
    SOBOL = "SOBOL"


class STANDARD_KERNEL(object):
    """
    Analytically defined kernels (includes BIAS and LINEAR)
    """
    RBF = "RBF"
    M12 = "M12"
    M32 = "M32"
    M52 = "M52"
    LINEAR = "LINEAR"
    BIAS = "BIAS"


class METRIC(object):
    FROBENIUS_NORMALISED = "FROBENIUS_NORMALISED"
