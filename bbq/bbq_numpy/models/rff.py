from bbq.bbq_numpy.utils.enums import QMC_SEQUENCE, QMC_SCRAMBLING, \
    QMC_KWARG, STANDARD_KERNEL
from bbq.bbq_numpy.models import rff_transformers, rff_samplers
from bbq.bbq_numpy.models.rff_samplers import QMCSequence
from bbq.bbq_numpy.utils.math import cartesian_product
import numpy as np


class FComposition(object):
    """
    Feature Composition class
    """

    def __init__(self, f_dict, composition):
        self.f_dict = f_dict
        self.composition = composition
        self.m = self.get_dim()

    def transform(self, x):
        phi_dict = {key: self.f_dict[key].transform(x) for key in
                    list(self.f_dict.keys())}
        phi = None
        k_id = self.composition[0]
        phi_temp = [phi_dict[k_id]]
        for i in range(1, self.composition.__len__(), 2):
            op = self.composition[i]
            k_id = self.composition[i + 1]
            if op == "*":
                phi_temp.append(phi_dict[k_id])
            elif op == "+":
                if phi is None:
                    phi = cartesian_product(phi_temp)
                else:
                    phi = np.hstack((phi, cartesian_product(phi_temp)))
                phi_temp = [phi_dict[k_id]]
            if i == self.composition.__len__() - 2:  # At the end. Add last
                if phi is None:
                    phi = cartesian_product(phi_temp)
                else:
                    phi = np.hstack((phi, cartesian_product(phi_temp)))
        if self.composition.__len__() == 1:
            phi = phi_temp[0]
        return phi

    def get_dim(self):
        m = 0
        k_id = self.composition[0]
        m_temp = self.f_dict[k_id].get_dim()
        for i in range(1, self.composition.__len__(), 2):
            op = self.composition[i]
            k_id = self.composition[i + 1]
            if op == "*":
                m_temp *= self.f_dict[k_id].get_dim()
            elif op == "+":
                m += m_temp
                m_temp = self.f_dict[k_id].get_dim()
            if i == self.composition.__len__() - 2:  # At the end. Add last
                m += m_temp
        if self.composition.__len__() == 1:
            m = m_temp
        return m


class Basis:
    """
    Base feature map class
    """

    def __init__(self):
        self.half_m = None
        self.m = None

    def get_dim(self):
        """
        Get the output dimensionality of this basis.
        This simply uses the instance's own get_dim() function
        """
        if not hasattr(self, 'm'):
            self.m = None
        return self.m


class Linear(Basis):
    def __init__(self, d, has_bias_term=True, transformer=rff_transformers.linear):
        Basis.__init__(self)
        self.d = d
        self.has_bias_term = has_bias_term
        self.__transformer = transformer

    def transform(self, x):
        """
        Applies the instance's sample weight expansion function to some data X
        h is the constant magnitude scalar
        :param x: The input data we want to transform
        :return: The fourier represented feature map for this particular kernel
        """
        return self.__transformer(x, self.has_bias_term)

    def get_dim(self):
        """
        X_shape is the shape tuple.
        For the linear basis we only care about the [1] which is
        the dimensionality
        :return:
        """
        if not hasattr(self, "m"):
            if self.has_bias_term is True:
                self.m = self.d + 1
            else:
                self.m = self.d
        return self.m


class LengthscaleBasis(Basis):
    # Cached
    def get_dim(self):
        return self.m


class RFF_RBF(LengthscaleBasis):
    def __init__(self, m, d, ls=1.0, sampler=rff_samplers.RFF_RBF,
                 transformer=rff_transformers.cos_sin):
        """
        :param m:   int
                    The final dimensionality of our features. M=2m
        :param d:   int
                    The dimensionality of our input data
        :param sampler: The sampling function
        :param transformer: The transformer function
        """
        LengthscaleBasis.__init__(self)
        if m % 2 != 0:
            raise ValueError("M must be an even number")
        self.m = m
        self.half_m = m // 2  # This is half the number of features
        self.d = d
        self.s = None  # The spectral weights
        self.ls = ls
        self.__sampler = sampler  # The sampling function. e.g. FSF_RBF
        # The basis transformation function. e.g. cos_sin_ssff
        self.__transformer = transformer
        # This is defined in the actual feature mapper (e.g. RFF_RBF())
        self.__sample_weights()

    def __sample_weights(self):
        """
        Allows one to resamples the internal spectral weights
        This would typically occur after an optimisation step
        The lengthscale can be optimized separately
        Assumes self.params order is known apriori
        I.e. the sampler and transformer should match each other
        """
        self.s = self.__sampler(d=self.d, m=self.half_m)

    def transform(self, x):
        """
        Applies the instance's sample weight expansion function to some data X
        h is the constant magnitude scalar
        :param x: The input data we want to transform
        :return: The fourier represented feature map for this particular kernel
        """
        return self.__transformer(x / self.ls, s=self.s)


class QMCF(LengthscaleBasis):
    """
    Quasi-Monte-Carlo (Fourier) Features for standard shift invariant kernels
    """

    def __init__(self, m, d, ls=1.0,
                 kernel_type=STANDARD_KERNEL.RBF,
                 sequence_type=QMC_SEQUENCE.HALTON,
                 scramble_type=QMC_SCRAMBLING.GENERALISED,
                 qmckwargs={QMC_KWARG.PERM: None},
                 sampler=rff_samplers.QMCF,
                 transformer=rff_transformers.cos_sin):
        """
        :param m:   int
                    The final dimensionality of our features. M=2m
        :param d:   int
                    The dimensionality of our input data
        :param sampler: The sampling function
        :param transformer: The transformer function
        """
        LengthscaleBasis.__init__(self)
        if m % 2 != 0:
            raise ValueError("M must be an even number")
        self.m = m
        self.half_m = m // 2  # This is half the number of features
        self.d = d
        self.kernel_type = kernel_type
        self.sequencer_type = sequence_type
        self.scramble_type = scramble_type
        self.qmc_kwargs = qmckwargs
        self.sequence = QMCSequence(n=self.half_m, d=self.d,
                                    sequence_type=self.sequencer_type,
                                    scramble_type=self.scramble_type,
                                    qmc_kwargs=self.qmc_kwargs)
        self.s = None  # The spectral weights
        self.ls = ls
        self.__sampler = sampler  # The sampling function. e.g. FSF_RBF
        # The basis transformation function.e.g. cos_sin_ssff
        self.__transformer = transformer
        # This is defined in the actual feature mapper (e.g. RFF_RBF())
        self.__sample_weights()

    def __sample_weights(self):
        """
        Allows one to resamples the internal spectral weights
        This would typically occur after an optimisation step
        The lengthscale can be optimized separately
        Assumes self.params order is known apriori
        I.e. the sampler and transformer should match each other
        """
        self.s = self.__sampler(sequence=self.sequence,
                                kernel_type=self.kernel_type)

    def transform(self, x):
        """
        Applies the instance's sample weight expansion function to some data X
        h is the constant magnitude scalar
        :param x: The input data we want to transform
        :return: The fourier represented feature map for this particular kernel
        """
        return self.__transformer(x / self.ls, s=self.s)


class QMCF_BBQ(LengthscaleBasis):
    """
    Black Box Quantile Quasi-Monte-Carlo (Fourier) Features
    for approximating valid black-box kernels
    """

    def __init__(self, m, d, sampler,
                 sequence_type=QMC_SEQUENCE.HALTON,
                 scramble_type=QMC_SCRAMBLING.GENERALISED,
                 qmc_kwargs={QMC_KWARG.PERM: None},
                 transformer=rff_transformers.cos_sin):
        """
        :param m:   int
                    The final dimensionality of our features. M=2m
        :param d:   int
                    The dimensionality of our input data
        :param sampler: The sampling function
        :param transformer: The transformer function
        """
        LengthscaleBasis.__init__(self)
        if m % 2 != 0:
            raise ValueError("M must be an even number")
        self.m = m
        self.half_m = m // 2  # This is half the number of features
        self.d = d
        self.sequencer_type = sequence_type
        self.scramble_type = scramble_type
        self.qmc_kwargs = qmc_kwargs
        self.sequence = QMCSequence(n=self.half_m, d=self.d,
                                    sequence_type=self.sequencer_type,
                                    scramble_type=self.scramble_type,
                                    qmc_kwargs=self.qmc_kwargs)
        self.s = None  # The spectral weights
        self.__sampler = sampler  # The sampling function. e.g. FSF_RBF
        # The basis transformation function. e.g. cos_sin_ssff
        self.__transformer = transformer
        # This is defined in the actual feature mapper (e.g. RFF_RBF())
        self.__sample_weights()

    def __sample_weights(self):
        """
        Allows one to resamples the internal spectral weights
        This would typically occur after an optimisation step
        The lengthscale can be optimized separately
        Assumes self.params order is known apriori
        I.e. the sampler and transformer should match each other
        """
        self.s = self.__sampler(sequence=self.sequence)

    def transform(self, x):
        """
        Applies the instance's sample weight expansion function to some data X
        h is the constant magnitude scalar
        :param x: The input data we want to transform
        :return: The fourier represented feature map for this particular kernel
        """
        return self.__transformer(x, s=self.s)


class QMCF_BBQ_ARD(LengthscaleBasis):
    """
    Black Box Quantile Quasi-Monte-Carlo (Fourier) Features
    for approximating valid black-box kernels
    """

    def __init__(self, m, d, sampler,
                 sequence_type=QMC_SEQUENCE.HALTON,
                 scramble_type=QMC_SCRAMBLING.GENERALISED,
                 qmckwargs={QMC_KWARG.PERM: None},
                 transformer=rff_transformers.cos_sin):
        """
        :param m:   int
                    The final dimensionality of our features. M=2m
        :param d:   int
                    The dimensionality of our input data
        :param sampler: The sampling function
        :param transformer: The transformer function
        """
        LengthscaleBasis.__init__(self)
        if m % 2 != 0:
            raise ValueError("M must be an even number")
        self.m = m
        self.half_m = m // 2  # This is half the number of features
        self.d = d
        self.sequencer_type = sequence_type
        self.scramble_type = scramble_type
        self.qmc_kwargs = qmckwargs
        self.sequence = QMCSequence(n=self.half_m, d=self.d,
                                    sequence_type=self.sequencer_type,
                                    scramble_type=self.scramble_type,
                                    qmc_kwargs=self.qmc_kwargs)
        self.s = None  # The spectral weights
        self.__sampler = sampler  # The sampling function. e.g. FSF_RBF
        # The basis transformation function. e.g. cos_sin_ssff
        self.__transformer = transformer
        # This is defined in the actual feature mapper (e.g. RFF_RBF())
        self.__sample_weights()

    def __sample_weights(self):
        """
        Allows one to resamples the internal spectral weights
        This would typically occur after an optimisation step
        The lengthscale can be optimized separately
        Assumes self.params order is known apriori
        I.e. the sampler and transformer should match each other
        """
        self.s = self.__sampler(sequence=self.sequence)

    def transform(self, x):
        """
        Applies the instance's sample weight expansion function to some data X
        h is the constant magnitude scalar
        :param x: The input data we want to transform
        :return: The fourier represented feature map for this particular kernel
        """
        return self.__transformer(x, s=self.s)
