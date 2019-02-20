from bbq.bbq_numpy.utils.enums import QMC_SEQUENCE, QMC_SCRAMBLING, \
    QMC_KWARG, STANDARD_KERNEL
import ghalton as gh
import bbq.bbq_numpy.models.rff_quantiles as qfs
import numpy as np


def QMCF_BBQ(sequence, interp_func):
    if isinstance(interp_func, list):
        d = sequence.points.shape[1]
        s = np.concatenate(
            [interp_func[d](sequence.points[:, [d]]) for d in range(d)],
            axis=1)
    else:
        s = interp_func(sequence.points)
    return s.T


def QMCF(sequence, kernel_type=STANDARD_KERNEL.RBF):
    """
    General wrapper for all standard Quasi-Monte Carlo (Fourier) Features
    :param sequence:     QMCSequence object
    :param kernel_type:  enum
    :return:
    """
    quantile_lut = {STANDARD_KERNEL.RBF: qfs.standard_normal,
                    STANDARD_KERNEL.M12: qfs.matern_12,
                    STANDARD_KERNEL.M32: qfs.matern_32,
                    STANDARD_KERNEL.M52: qfs.matern_52}
    qf = quantile_lut[kernel_type]
    s = qf(sequence.points)
    return s.T


def RFF_RBF(d, m):
    """
    Sampling scheme as per (Rahimi, Recht 2007) for RBF Random Fourier Features
    :param d:
    :param m:
    :return: spectral frequencies
    """
    s = np.random.normal(size=(d, m))
    return s


class QMCSequence:
    def __init__(self, n, d, sequence_type=QMC_SEQUENCE.HALTON,
                 scramble_type=QMC_SCRAMBLING.GENERALISED,
                 qmc_kwargs={QMC_KWARG.PERM: None}):
        self.n = n
        self.d = d
        self.sequence_type = sequence_type
        self.scramble_type = scramble_type
        self.qmc_kwargs = qmc_kwargs
        self.sequencer = None
        self.points = None
        self.init_sequencer()
        self.init_sequence()  # Could make this autoinitialisation optional

    def init_sequencer(self):
        # ---------------------------------------#
        #                HALTON                  #
        # ---------------------------------------#
        if self.sequence_type == QMC_SEQUENCE.HALTON:
            if self.scramble_type == QMC_SCRAMBLING.GENERALISED:
                if self.qmc_kwargs[QMC_KWARG.PERM] is None:
                    perm = gh.EA_PERMS[:self.d]  # Default permutation
                else:
                    perm = self.qmc_kwargs[QMC_KWARG.PERM]
                self.sequencer = gh.GeneralizedHalton(perm)
            else:
                self.sequencer = gh.Halton(int(self.d))

    def init_sequence(self):
        self.points = np.array(self.sequencer.get(int(self.n)))

    def extend_sequence(self):
        pass
