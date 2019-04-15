import numpy as np
import tensorflow as tf


class BoundedQPoints():
    def __init__(self,
                 N,
                 p1=(0.025, -10.0),
                 p2=(0.975, 10.0),
                 trainable=True,
                 random_init=True,
                 tfdt=tf.float64):
        """

        :param N:   integer
                    Number of interpolating points

        :param p1:  tuple (xmin, ymin)
                    The bottom left quantile point before asymptote

        :param p2:  tuple (xmax, ymax)
                    The top right quantile point before asymptote
        :param trainable:
        :param random_init:
        :param tfdt:
        """
        self.N = N
        self.trainable = trainable
        self.tfdt = tfdt
        self.p1x = tf.sigmoid(tf.Variable(p1[0], dtype=self.tfdt)) / 2.0 - 1e-8
        self.DELTA_x2 = tf.sigmoid(
            tf.Variable(p2[0], dtype=self.tfdt)) / 2.0 - 1e-8
        self.DELTA_xmid = 1.0 - self.DELTA_x2 - self.p1x - 1e-8
        self.p1y = tf.Variable(p1[1], dtype=tfdt)
        self.p2y = self.p1y + tf.nn.softplus(
            tf.Variable(p2[1], dtype=tfdt))
        self.DELTA_ymid = self.p2y - self.p1y
        self.DELTAS = tf.reshape(
            tf.stack([self.DELTA_xmid, self.DELTA_ymid], axis=0), (1, -1))
        self.p1 = tf.reshape(tf.stack([self.p1x, self.p1y], axis=0), (1, -1))
        self.p2 = tf.reshape(
            tf.stack([self.p1x + self.DELTA_xmid, self.p2y], axis=0), (1, -1))

        # 2D offset coordinates
        if random_init:
            self.offsets = [
                tf.sigmoid(
                    tf.Variable(
                        initial_value=np.random.uniform(-2.0, 2.0, size=(1, 2)),
                        trainable=trainable, dtype=tfdt)) for _ in
                range(self.N - 1)]
        else:
            self.offsets = [tf.clip_by_value(
                tf.Variable(initial_value=np.ones(shape=(1, 2)),
                            trainable=trainable, dtype=tfdt),
                -1.0, 1.0) for _ in
                range(self.N - 1)]
        self.sum_all_offsets = tf.add_n(
            [tf.abs(v) for v in self.offsets])  # dx and dy sums
        self.rescale_factor = self.sum_all_offsets / self.DELTAS
        self.point_ops = [self.p1] + \
                         [self.p1 + self.offset_sum(N=i) / self.rescale_factor
                          for i in
                          range(1, self.N)]
        self.points = tf.concat(self.point_ops, axis=0)

    def offset_sum(self, N):
        return tf.add_n([tf.abs(v) for v in self.offsets[0:N]])
