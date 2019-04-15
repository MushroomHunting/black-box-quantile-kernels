import tensorflow as tf
from ..com.enums import QMC_SEQUENCE, QMC_SCRAMBLING, QMC_KWARG, STANDARD_KERNEL
from .. import samplers, transformers
from .qmc import QMCSequence
from ..utils.math import cartesian_product

class FComposition(object):
    """
    Feature Composition class
    """

    def __init__(self, f_dict, composition, tfdt=tf.float64):
        self.f_dict = f_dict
        self.composition = composition
        self.tfdt = tfdt
        self.M = self.get_dim()

    def transform(self, X):
        phi_dict = {key: self.f_dict[key].transform(X) for key in
                    list(self.f_dict.keys())}
        phi = tf.zeros(shape=X.shape[0],
                       dtype=self.tfdt)  # TODO Make this a global?
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
                    phi = tf.concat([phi, cartesian_product(phi_temp)], axis=1)
                phi_temp = [phi_dict[k_id]]
            if i == self.composition.__len__() - 2:  # We're at the end. Add last
                if phi is None:
                    phi = cartesian_product(phi_temp)
                else:
                    phi = tf.concat([phi, cartesian_product(phi_temp)], axis=1)
        if self.composition.__len__() == 1:
            phi = phi_temp[0]
        return phi

    def get_dim(self):
        M = 0
        k_id = self.composition[0]
        M_temp = self.f_dict[k_id].get_dim()
        for i in range(1, self.composition.__len__(), 2):
            op = self.composition[i]
            k_id = self.composition[i + 1]
            if op == "*":
                M_temp *= self.f_dict[k_id].get_dim()
            elif op == "+":
                M += M_temp
                M_temp = self.f_dict[k_id].get_dim()
            if i == self.composition.__len__() - 2:  # We're at the end. Add last
                M += M_temp
        if self.composition.__len__() == 1:
            M = M_temp
        return M


class Basis(object):
    """
    Base feature map class
    """

    def get_dim(self):
        """
        Get the output dimensionality of this basis.
        This simply uses the instance's own get_dim() function
        """
        if not hasattr(self, 'M'):
            self.M = None
        return self.M


class Linear(Basis):
    def __init__(self,
                 D,
                 has_bias_term=True,
                 transformer=transformers.linear,
                 tfdt=tf.float64):
        super().__init__()
        self.D = D,
        self.tfdt = tfdt
        self.has_bias_term = has_bias_term
        self.__transformer = transformer

    def transform(self, X):
        """
        Applies the instance's sample weight expansion function to some data X
        h is the constant magnitude scalar
        :param X: The input data we want to transform
        :return: The feature map
        """
        return self.__transformer(X, self.has_bias_term, tfdt=self.tfdt)

    def get_dim(self):
        """
        X_shape is the shape tuple.
        For the linear basis we only care about the [1] which is
        the dimensionality
        :return:
        """
        if not hasattr(self, "M"):
            if self.has_bias_term is True:
                self.M = self.D + 1
            else:
                self.M = self.D
        return self.M


class Lengthscale_Basis(Basis):
    def __init__(self):
        super().__init__()

    def get_dim(self):
        return self.M


class RFF_RBF(Lengthscale_Basis):
    def __init__(self,
                 M,
                 D,
                 ls=1.0,
                 sampler=samplers.RFF_RBF,
                 transformer=transformers.cos_sin,
                 tfdt=tf.float64):
        """
        :param M:   int
                    Feature space dimensionality
        :param D:   int
                    The dimensionality of our input data
        :param sampler: The sampling function
        :param transformer: The transformer function
        """
        super().__init__()
        self.tfdt = tfdt
        if M % 2 != 0:
            raise ValueError("M must be an even number")
        self.M = M
        self.m = M // 2  # This is half the number of features
        self.D = D
        self.S = None  # The spectral weights

        self.ls = tf.Variable(ls, self.tfdt)
        self.__sampler = sampler  # The sampling function
        self.__transformer = transformer  # The basis transformation function. e.g. cos_sin
        self.__sample_weights()

    def __sample_weights(self):
        """
        Allows one to resamples the internal spectral weights
        The lengthscale can be optimized separately for lengthscale kernels
        """
        self.S = self.__sampler(D=self.D, m=self.m, tfdt=self.tfdt)

    def transform(self, X):
        """
        Applies the instance's sample weight expansion function to some data X
        :param X: The input data we want to transform
        :return: The fourier represented feature map for this particular kernel
        """
        return self.__transformer(tf.divide(X, self.ls), S=self.S,
                                  tfdt=self.tfdt)


class QMCF(Lengthscale_Basis):
    """
    Quasi-Monte-Carlo (Fourier) Features for standard shift invariant kernels
    """

    def __init__(self,
                 M,
                 D,
                 ls=1.0,
                 kernel_type=STANDARD_KERNEL.RBF,
                 sequence_type=QMC_SEQUENCE.HALTON,
                 scramble_type=QMC_SCRAMBLING.OWEN17,
                 qmckwargs={QMC_KWARG.PERM: None},
                 sampler=samplers.QMCF,
                 transformer=transformers.cos_sin,
                 tfdt=tf.float64):
        """
        :param M:   int
                    The base dimensionality of our features.
                    instead of approximate sqrt(2)*cos(wx+b) representation
        :param D:   int
                    The dimensionality of our input data
        :param ls:  Lengthscale
        :param sampler: The sampling function
        :param transformer: The transformer function
        """
        super().__init__()
        self.tfdt = tfdt
        if M % 2 != 0:
            raise ValueError("M must be an even number")
        self.M = M
        self.m = M // 2  # This is half the number of features
        self.D = D
        self.kernel_type = kernel_type
        self.sequencer_type = sequence_type
        self.scramble_type = scramble_type
        self.qmckwargs = qmckwargs
        self.sequence = QMCSequence(N=self.m, D=self.D,
                                    sequence_type=self.sequencer_type,
                                    scramble_type=self.scramble_type,
                                    qmckwargs=self.qmckwargs,
                                    tfdt=self.tfdt)

        self.S = None  # The spectral weights
        self.ls = tf.nn.softplus(tf.Variable(initial_value=ls, dtype=self.tfdt))
        self.__sampler = sampler  # The sampling function.
        self.__transformer = transformer  # The basis transformation function. e.g. cos_sin
        self.__sample_weights()  # This is defined in the actual feature mapper (e.g. RFF_RBF())

    def __sample_weights(self):
        """
        Allows one to resample the internal spectral weights
        """
        self.S = self.__sampler(sequence=self.sequence,
                                kernel_type=self.kernel_type, tfdt=self.tfdt)

    def transform(self, X):
        """
        Applies the instance's sample weight expansion function to some data X
        :param X: The input data we want to transform
        :return: The fourier represented feature map for this particular kernel
        """
        return self.__transformer(
            tf.divide(X, self.ls),
            S=self.S,
            tfdt=self.tfdt)


class QMCF_BBQ(Lengthscale_Basis):
    """
    Black Box Quantile Quasi-Monte-Carlo (Fourier) Features
    for approximating valid black-box kernels
    """

    def __init__(self,
                 M,
                 D,
                 bqp,
                 sampler,
                 sequence_type=QMC_SEQUENCE.HALTON,
                 scramble_type=QMC_SCRAMBLING.OWEN17,
                 qmckwargs={QMC_KWARG.PERM: None},
                 transformer=transformers.cos_sin,
                 tfdt=tf.float64):
        """
        :param M:   int
                    The final dimensionality of our features. M=2m
        :param D:   int
                    The dimensionality of our input data
        :param bqp: BoundedQPoints
                    The point parametrisation
        :param sampler: The sampling function
        :param transformer: The transformer function
        """
        super().__init__()
        self.tfdt = tfdt
        if M % 2 != 0:
            raise ValueError("M must be an even number")
        self.M = M
        self.m = M // 2  # This is half the number of features
        self.D = D
        self.bqp = bqp
        self.sequencer_type = sequence_type
        self.scramble_type = scramble_type
        self.qmckwargs = qmckwargs
        self.sequence = QMCSequence(N=self.m, D=self.D,
                                    sequence_type=self.sequencer_type,
                                    scramble_type=self.scramble_type,
                                    qmckwargs=self.qmckwargs,
                                    tfdt=self.tfdt)
        self.S = None  # The spectral weights
        self.__sampler = sampler  # The sampling function.
        self.__transformer = transformer  # The basis transformation function. e.g. cos_sin
        self.__sample_weights()  # This is defined in the actual feature mapper (e.g. RFF_RBF())

    def __sample_weights(self):
        """
        Allows one to resamples the internal spectral weights
        """
        self.S = self.__sampler(x_query=self.sequence.points)

    def transform(self, X):
        """
        Applies the instance's sample weight expansion function to some data X
        :param X: The input data we want to transform
        :return: The fourier represented feature map for this particular kernel
        """
        return self.__transformer(X, S=self.S, tfdt=self.tfdt, m=self.m)

class QMCF_BBQ_ARD(Lengthscale_Basis):
    """
    Black Box Quantile Quasi-Monte-Carlo (Fourier) Features
    for approximating valid black-box kernels
    """

    def __init__(self,
                 M,
                 D,
                 bqp1,
                 bqp2,
                 sampler1,
                 sampler2,
                 sequence_type=QMC_SEQUENCE.HALTON,
                 scramble_type=QMC_SCRAMBLING.OWEN17,
                 qmckwargs={QMC_KWARG.PERM: None},
                 transformer=transformers.cos_sin_ard,
                 tfdt=tf.float64):
        """
        :param M:   int
                    The final dimensionality of our features. M=2m
        :param D:   int
                    The dimensionality of our input data
        :param sampler: The sampling function
        :param transformer: The transformer function
        """
        super().__init__()
        self.tfdt = tfdt
        if M % 2 != 0:
            raise ValueError("M must be an even number")
        self.M = M
        self.m = M // 2  # This is half the number of features
        self.D = D
        self.bqp1 = bqp1
        self.bqp2 = bqp2
        self.sequencer_type = sequence_type
        self.scramble_type = scramble_type
        self.qmckwargs = qmckwargs
        self.sequence = QMCSequence(N=self.m, D=self.D,
                                    sequence_type=self.sequencer_type,
                                    scramble_type=self.scramble_type,
                                    qmckwargs=self.qmckwargs,
                                    tfdt=self.tfdt)
        self.S = None  # The spectral weights
        self.__sampler1 = sampler1  # The sampling function (dim 1)
        self.__sampler2 = sampler2  # The sampling function (dim 2)
        self.__transformer = transformer  # The basis transformation function. e.g. cos_sin
        self.__sample_weights()  # This is defined in the actual feature mapper (e.g. RFF_RBF())

    def __sample_weights(self):
        """
        Allows one to resamples the internal spectral weights
        """
        self.S1 = self.__sampler1(x_query=self.sequence.points)
        self.S2 = self.__sampler2(x_query=self.sequence.points)

    def transform(self, X):
        """
        Applies the instance's sample weight expansion function to some data X
        :param X: The input data we want to transform
        :return: The fourier represented feature map for this particular kernel
        """
        return self.__transformer(X, S1=self.S1, S2=self.S2, tfdt=self.tfdt, m=self.m)