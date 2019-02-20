from bbq.bbq_numpy.utils.math import smw_inv_correction
from bbq.bbq_numpy.utils.batch import batch_generator
from bbq.bbq_numpy.utils.containers import Prediction
from tqdm import tqdm
import numpy as np
from time import time


class ABLR(object):
    """
    Purely analytic BLR with rank-k streaming updates
    """

    def __init__(self, k_phi, d, d_out=1, alpha=1.0, beta=1.0, log_dir=None,
                 verbose=0):
        self.N_trn = None  # The total number of training samples to stream
        # through
        self.d_out = d_out
        self.m = None  # Total number of features in the complete feature
        # expansion
        self.log_dir = log_dir
        self.verbose = verbose

        self.k_phi = k_phi
        self.m = self.k_phi.get_dim()  # Fully expanded composition
        # dimensionality
        self.alpha = alpha
        self.beta = beta

        self.s_inv_tgt = np.zeros(shape=(self.m, 1))  # Required weight update
        self.s = np.identity(self.m) / self.alpha  # a.k.a. a_inv
        self.s_inv = np.identity(self.m) * self.alpha  # a.k.a. A
        self.phi = None
        self.mu = None
        self.y = None

    def update(self, x, y):
        self.phi = self.k_phi.transform(x)
        self.s -= smw_inv_correction(a_inv=self.s,
                                     u=np.sqrt(self.beta) * self.phi.T,
                                     v=np.sqrt(self.beta) * self.phi)

        self.s_inv_tgt += self.beta * np.dot(self.phi.T, y)
        self.s_inv += self.beta * np.dot(self.phi.T, self.phi)
        self.mu = np.dot(self.s, self.s_inv_tgt)
        self.y = np.dot(self.phi, self.mu)

    def learn_from_history(self, x_trn, y_trn, batch_size=None):
        # Define the batch data generator. This maintains an internal counter
        # and also allows wraparound for multiple epochs

        # Compute optimal batch size
        if batch_size is None:
            batch_size = int(np.cbrt(self.m ** 2 / 2))
            if self.verbose > 0:
                print("Batch size is {}".format(batch_size))

        data_batch = batch_generator(arrays=[x_trn, y_trn],
                                     batch_size=batch_size,
                                     wrap_last_batch=False)

        n = x_trn.shape[0]  # Alias for the total number of training samples
        n_batches = int(np.ceil(n / batch_size))  # The number of batches

        """ Run the batched inference """
        t1 = time()
        if self.verbose > 0:
            for _ in tqdm(range(n_batches)):
                x_batch, y_batch = next(data_batch)
                self.update(x=x_batch, y=y_batch)
        else:
            for _ in range(n_batches):
                x_batch, y_batch = next(data_batch)
                self.update(x=x_batch, y=y_batch)

        if self.verbose > 0:
            lap_time = time() - t1
            print("Time taken for learning: {} s".format(np.round(lap_time, 5)))

    def predict(self, x_tst, pred_var=True):
        """
        https://discourse.edwardlib.org/t/how-to-obtain-prediction-results/215
        :param x_tst:
        :param pred_var:        Boolean
        :return:
        """
        # Initialise a prediction result object to dump results into
        prediction = Prediction()
        phi = self.k_phi.transform(x_tst)

        # Predict mean
        t1 = time()
        y_pred = np.dot(phi, self.mu)
        prediction.mean = np.atleast_2d(y_pred)
        time_taken = time() - t1
        if self.verbose > 0:
            print("Got pred mean for {} points in {} s".format(
                prediction.mean.shape, np.round(time_taken, 5)))

        # Predict var
        if pred_var:
            t1 = time()
            prediction.var = np.sum(
                np.dot(phi, self.s) * phi, axis=1, keepdims=True)
            if self.verbose > 0:
                time_taken = time() - t1
                print("Got pred var for {} points in {} s".format(
                    prediction.var.shape, np.round(time_taken, 5)))

        return prediction
