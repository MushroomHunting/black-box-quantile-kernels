from ..utils.math import smw_inv_precomp, logdet_correction_precomp, \
    smw_nlml as nlml_loss
from ..com.containers import Prediction
from ..utils.inference import batch_generator
from tqdm import tqdm
import tensorflow as tf


class ABLR(object):
    """
    Batched analytic BLR with SGD for hyperparameter optimimsation
    """

    def __init__(self,
                 kphi,
                 N_trn,
                 batch_size,
                 D,
                 D_out,
                 alpha=1.0,
                 beta=1.0,
                 tfdt=tf.float64,
                 learning_rate=1e-1,
                 adam_beta1=0.8,
                 adam_beta2=0.999,
                 adam_eps=1e-9,
                 verbose=0,
                 model_scope="ABLR"):
        self.tfdt = tfdt
        self.verbose = verbose
        self.N_trn = N_trn
        self.batch_size = batch_size
        self.D = D
        self.D_out = D_out
        self.X_plh = tf.placeholder(dtype=self.tfdt, shape=[None, self.D],
                                    name="X_plh")
        self.Y_plh = tf.placeholder(dtype=self.tfdt, shape=[None, self.D_out],
                                    name="Y_plh")
        self.model_scope = model_scope
        self.kphi = kphi
        self.M = self.kphi.get_dim()
        self.alpha = tf.nn.softplus(tf.Variable(alpha, dtype=self.tfdt))
        self.beta = tf.nn.softplus(tf.Variable(beta, dtype=self.tfdt))
        self.Sinv_tgt = tf.Variable(
            tf.zeros(shape=[self.M, self.D_out], dtype=self.tfdt),
            trainable=False,
            dtype=self.tfdt)
        self.S = tf.Variable(
            tf.eye(num_rows=self.M, dtype=self.tfdt) / self.alpha,
            trainable=False, dtype=self.tfdt)

        # Initial value of 'streaming' (batch) log determinant
        # log|alpha*I| = trace(log(alpha*I)) = M * log(alpha)
        self.Sinv_logdet = tf.Variable(self.M * tf.log(self.alpha),
                                       trainable=False, dtype=self.tfdt)
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
        #:::::::::::::::::::::::::::: RESET ops :::::::::::::::::::::::::::#
        self.Sinv_tgt_RESET = tf.assign(self.Sinv_tgt,
                                        tf.zeros(shape=[self.M, 1],
                                                 dtype=self.tfdt))
        self.S_RESET = tf.assign(self.S,
                                 tf.eye(num_rows=self.M,
                                        dtype=self.tfdt) / self.alpha)

        self.Sinv_logdet_RESET = tf.assign(self.Sinv_logdet,
                                           self.M * tf.log(self.alpha))
        #:::::::::::::::::::::::::::: RESET ops :::::::::::::::::::::::::::#
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

        # Define the model
        with tf.name_scope(self.model_scope):
            self.PHI = self.kphi.transform(self.X_plh)
            self.V = tf.sqrt(self.beta) * self.PHI
            self.VS = tf.matmul(self.V, self.S)
            rank = tf.shape(self.V)[0]
            self.IVSU = tf.eye(rank, dtype=tfdt) + tf.matmul(self.VS, self.V,
                                                             transpose_b=True)
            self.IVSU_R = tf.cholesky(self.IVSU)
            self.S_UPDATE = smw_inv_precomp(A_inv=self.S,
                                            VS=self.VS,
                                            R_lower=self.IVSU_R,
                                            tfdt=self.tfdt)

            self.Sinv_tgt_UPDATE = tf.assign_add(self.Sinv_tgt,
                                                 self.beta * tf.matmul(self.PHI,
                                                                       self.Y_plh,
                                                                       transpose_a=True))
            self.Sinv_logdet_UPDATE = logdet_correction_precomp(
                self.Sinv_logdet,
                R_lower=self.IVSU_R)
            self.mu = tf.matmul(self.S, self.Sinv_tgt)
            self.Y = tf.matmul(self.PHI, self.mu)

        self.nlml_loss = nlml_loss(Y_pred=self.Y,
                                   mu=self.mu,
                                   N=self.N_trn,
                                   M=self.M,
                                   Y_trn=self.Y_plh,
                                   alpha=self.alpha,
                                   beta=self.beta,
                                   logdet=self.Sinv_logdet,
                                   tfdt=self.tfdt)

        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        self.train_step = tf.train.AdamOptimizer(self.learning_rate,
                                                 beta1=self.adam_beta1,
                                                 beta2=self.adam_beta2,
                                                 epsilon=self.adam_eps).minimize(
            self.nlml_loss)

        self.sess = tf.Session()
        with self.sess.as_default():
            assert self.sess is tf.get_default_session()

    def reset_blr_vars(self):
        """
        Resets all blr variables to their default
        :return:
        """
        _, _, _ = self.sess.run(
            [self.Sinv_tgt_RESET,
             self.S_RESET,
             self.Sinv_logdet_RESET])

    def update(self, feed_dict):
        _, _, _ = self.sess.run(
            fetches=[self.S_UPDATE,
                     self.Sinv_tgt_UPDATE,
                     self.Sinv_logdet_UPDATE],
            feed_dict=feed_dict)

    def learn_from_history(self,
                           X_trn,
                           Y_trn):

        data_batch = batch_generator(arrays=[X_trn, Y_trn],
                                     batch_size=self.batch_size)
        n_batches = 1 + self.N_trn // self.batch_size

        for i in range(n_batches):
            X_batch, Y_batch = next(data_batch)
            X_trn_feeddict = {self.X_plh: X_batch}
            Y_feeddict = {self.Y_plh: Y_batch,
                          }
            batch_feeddict = {
                **X_trn_feeddict,
                **Y_feeddict,
            }

            self.update(feed_dict=batch_feeddict)

    def learn_hypers(self,
                     X_trn,
                     Y_trn,
                     n_epochs=1):
        self.steps = n_epochs
        self.sess.run(tf.global_variables_initializer())
        X_trn_feeddict = {self.X_plh: X_trn}
        Y_trn_feeddict = {self.Y_plh: Y_trn}
        trn_feeddict = {**X_trn_feeddict,
                        **Y_trn_feeddict,
                        }
        for i in tqdm(range(self.steps)):
            self.learn_from_history(X_trn=X_trn, Y_trn=Y_trn)

            # 2. Run the training loss
            _, step_loss = self.sess.run([self.train_step,
                                          self.nlml_loss],
                                         feed_dict=trn_feeddict)
            print("\nLOSS: {}".format(step_loss))

            # 3. Reset the blr variables
            if i != self.steps - 1:
                self.reset_blr_vars()

    def predict(self,
                X_tst,
                pred_var=True,
                ):
        """
        :param X_tst:
        :param pred_var:        Boolean
        :return:
        """
        prediction = Prediction()
        X_tst_feeddict = {self.X_plh: X_tst}
        mc_mean = tf.reduce_mean(self.Y, 1, keepdims=True)
        prediction.mean = self.sess.run(mc_mean, feed_dict=X_tst_feeddict)

        if pred_var:
            var_pred_OP = 1 / self.beta + tf.reduce_sum(
                tf.multiply(tf.matmul(self.PHI, self.S), self.PHI), axis=1,
                keepdims=True)
            prediction.var = self.sess.run(fetches=var_pred_OP,
                                           feed_dict=X_tst_feeddict)
        return prediction
