import numpy as np
import ghalton as gh
from ..com.enums import QMC_SCRAMBLING, QMC_SEQUENCE, QMC_KWARG
import tensorflow as tf
import tensorflow_probability as tfp


class QMCSequence(object):
    def __init__(self,
                 N,
                 D,
                 seed=42,
                 sequence_type=QMC_SEQUENCE.HALTON,
                 scramble_type=QMC_SCRAMBLING.OWEN17,
                 qmckwargs={QMC_KWARG.PERM: None},
                 tfdt=tf.float64
                 ):
        self.tfdt = tfdt
        self.N = N
        self.D = D
        self.seed = seed
        self.sequence_type = sequence_type
        self.scramble_type = scramble_type
        self.qmckwargs = qmckwargs
        self.sequencer = None
        self.points = None
        self.init_sequence()

    def init_sequence(self):
        if self.sequence_type == QMC_SEQUENCE.HALTON:
            if self.scramble_type == QMC_SCRAMBLING.OWEN17:
                self.points = tf.Variable(
                    initial_value=tfp.mcmc.sample_halton_sequence(dim=self.D,
                                                                  num_results=self.N,
                                                                  dtype=self.tfdt,
                                                                  randomized=True,
                                                                  seed=self.seed),
                    dtype=self.tfdt,
                    trainable=False)
            elif self.scramble_type == QMC_SCRAMBLING.GENERALISED:
                if self.qmckwargs[QMC_KWARG.PERM] is None:
                    perm = gh.EA_PERMS[:self.D]  # Default permutation
                else:
                    perm = self.qmckwargs[QMC_KWARG.PERM]
                self.sequencer = gh.GeneralizedHalton(perm)
                self.points = tf.constant(
                    np.array(self.sequencer.get(int(self.N))), dtype=self.tfdt)
            else:
                self.sequencer = gh.Halton(int(self.D))
                self.points = tf.constant(
                    np.array(self.sequencer.get(int(self.N))), dtype=self.tfdt)

    def extend_sequence(self):
        pass
