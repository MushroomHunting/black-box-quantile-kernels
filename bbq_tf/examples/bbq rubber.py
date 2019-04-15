import numpy as np
import matplotlib
import matplotlib.pylab as plt
import seaborn as sbn
from functools import partial
from time import time
import tensorflow as tf
from matplotlib.patches import Rectangle

matplotlib.rcParams.update(
    {'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42,
     'legend.fontsize': 8, 'image.cmap': "viridis"})
sbn.set(font_scale=0.4)
sbn.set_context(rc={'lines.markeredgewidth': 0.25})
sbn.set_style("whitegrid")

from bbq_tf.utils.colour import named_CB as CBCOLS
from bbq_tf.models.ABLR import ABLR
from bbq_tf.features import fourier_features as ff
from bbq_tf.com.enums import QMC_SCRAMBLING
from bbq_tf.utils.parametrisations import BoundedQPoints
from bbq_tf.samplers import QMCF_BBQ as bbq_sampler
from bbq_tf.utils.interpolate import pchiptx_full, pchiptx_asymp
from bbq_tf.utils import datasets
from bbq_tf.utils.metrics import mnll, rmse

datasetName = "rubber"
root_dir = r"/home/datasets"
"""
Generate dataset
"""
if datasetName == "co2":
    train_data, test_data = datasets.mauna_loa(rootDir=root_dir)
elif datasetName == "airline":
    train_data, test_data = datasets.airline_passengers(rootDir=root_dir)
elif datasetName == "airfoil_noise":
    train_data, test_data = datasets.airfoil_noise(rootDir=root_dir)
elif datasetName == "concrete":
    train_data, test_data = datasets.concrete(rootDir=root_dir)
elif datasetName == "rubber" \
        or datasetName == "pores" \
        or datasetName == "tread":
    train_data, test_data = datasets.textures_2D(rootDir=root_dir,
                                                 texture_name=datasetName)

X_trn = train_data[:, :-1]
Y_trn = train_data[:, [-1]]
X_tst = test_data[:, :-1]
Y_tst = test_data[:, [-1]]

if __name__ == "__main__":
    tfdt = tf.float64
    THE_FIGURE = plt.figure(figsize=(8.25, 6.50), dpi=200)
    use_uniform_training = True
    D = 2
    do_learn_var = True
    M = 2000
    alpha = 20.0
    beta = 60
    res = 96
    D_out = 1
    N_trn = X_trn.shape[0]
    x_min = -1.0
    x_max = 1.0

    batch_size = 127
    batch_repeats = 1
    x_range = x_max - x_min
    x_min_tst = x_min
    x_max_tst = x_max

    x_trn_range = x_max_tst - x_min_tst
    y_trn_range = x_max_tst - x_min_tst

    x1_mesh_tst, x2_mesh_tst = np.meshgrid(
        np.linspace(x_min_tst, x_max_tst, res),
        np.linspace(x_min_tst, x_max_tst,
                    res))
    X_tst_visual = np.hstack(
        (x1_mesh_tst.reshape(-1, 1), x2_mesh_tst.reshape(-1, 1)))

    N_trn = X_trn.shape[0]
    ax = plt.subplot(3, 3, 1, xlim=(x_min_tst, x_max_tst),
                     ylim=(x_min_tst, x_max_tst))
    ax.scatter(X_trn[:, 0], X_trn[:, 1], Y_trn, cmap='grey')
    ax.set_xlim(x_min_tst, x_max_tst)
    ax.set_ylim(x_min_tst, x_max_tst)
    ax.add_patch(
        Rectangle((-0.5, -0.5), 1.0, 1.0, fill=None, alpha=1,
                  color=CBCOLS["darkgrey"]))

    ax = plt.subplot(3, 3, 2, xlim=(x_min_tst, x_max_tst),
                     ylim=(x_min_tst, x_max_tst))

    ax.set_xlim(x_min_tst, x_max_tst)
    ax.set_ylim(x_min_tst, x_max_tst)
    ax.add_patch(
        Rectangle((-0.5, -0.5), 1.0, 1.0, fill=None, alpha=1,
                  color=CBCOLS["darkgrey"]))

    """----------------------------
    |           Learning          |
    ----------------------------"""
    use_pchip_asymp = True

    N_STEPS = 225
    N_bqp = 25

    p1 = (np.random.uniform(-3, -2), np.random.uniform(-90, -60))
    p2 = (np.random.uniform(-3, -2), np.random.uniform(60, 90))

    p1B = (np.random.uniform(-3, -2), np.random.uniform(-90, -60))
    p2B = (np.random.uniform(-3, -2), np.random.uniform(60, 90))

    learning_rate = 0.5e-3
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_eps = 1e-10

    QP1 = BoundedQPoints(N_bqp,
                         p1=p1,
                         p2=p2,
                         trainable=True,
                         random_init=True,
                         tfdt=tfdt)
    QP2 = BoundedQPoints(N_bqp,
                         p1=p1B,
                         p2=p2B,
                         trainable=True,
                         random_init=True,
                         tfdt=tfdt)

    if use_pchip_asymp is True:
        my_sampler1 = partial(bbq_sampler, bqp_points=QP1.points,
                              qf=partial(pchiptx_asymp, tfdt=tfdt))
        my_sampler2 = partial(bbq_sampler, bqp_points=QP2.points,
                              qf=partial(pchiptx_asymp, tfdt=tfdt))
    else:
        my_sampler = partial(bbq_sampler, bqp_points=QP1.points,
                             qf=pchiptx_full)

    KPHI = ff.QMCF_BBQ_ARD(M=M, D=2,
                           bqp1=QP1,
                           bqp2=QP2,
                           sampler1=my_sampler1,
                           sampler2=my_sampler2,
                           scramble_type=QMC_SCRAMBLING.OWEN17,
                           tfdt=tfdt)
    # Define Bayesian Linear Regression class
    ABLR_learner = ABLR(kphi=KPHI,
                        N_trn=N_trn,
                        batch_size=batch_size,
                        D=2,  # 2*C, #D
                        D_out=D_out,
                        alpha=alpha,
                        beta=beta,
                        tfdt=tfdt)
    t1 = time()
    # Learn on a historical dataset
    ABLR_learner.learn_hypers(X_trn=X_trn,
                              Y_trn=Y_trn,
                              n_epochs=N_STEPS)
    time_taken = time() - t1
    print("Time taken: {}".format(time_taken))
    """------------------------------
    |           Prediction          |
    ------------------------------"""
    prediction_visual = ABLR_learner.predict(X_tst=X_tst_visual, pred_var=True)
    Y_predict_visual = prediction_visual.mean.reshape(x1_mesh_tst.shape)

    prediction = ABLR_learner.predict(X_tst=X_tst, pred_var=True)
    Y_predict = prediction.mean.reshape(Y_tst.shape)

    predict_metrics = {"rmse": rmse(y_actual=Y_tst, y_pred=prediction.mean),
                       "mnll": mnll(actual_mean=Y_tst,
                                    pred_mean=prediction.mean,
                                    pred_var=prediction.var)
                       }
    print("RMSE: {}\nMNLL: {}".format(predict_metrics["rmse"],
                                      predict_metrics["mnll"]))

    ax = plt.subplot(3, 3, 4)
    cs = plt.contourf(x1_mesh_tst, x2_mesh_tst, Y_predict_visual, res,
                      cmap="gray")
    ax.add_patch(
        Rectangle((-0.5, -0.5), 1.0, 1.0, fill=None, alpha=1,
                  color=CBCOLS["darkgrey"]))
    plt.colorbar(cs)
    ax.set_title("Predictive mean.\n M: {}".format(M))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(x_min_tst, x_max_tst)
    ax.set_ylim(x_min_tst, x_max_tst)

    if do_learn_var:
        Y_predict_var = prediction_visual.var.reshape(x1_mesh_tst.shape)
        ax = plt.subplot(3, 3, 5)
        cs = plt.contourf(x1_mesh_tst, x2_mesh_tst, Y_predict_var, res,
                          cmap="gray")
        ax.add_patch(
            Rectangle((-0.5, -0.5), 1.0, 1.0, fill=None, alpha=1,
                      color=CBCOLS["darkgrey"]))
        plt.colorbar(cs)
        ax.set_title("Predictive variance")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.xlim(x_min_tst, x_max_tst)
        plt.ylim(x_min_tst, x_max_tst)

    _XQUERY = np.linspace(0, 1, 2000)
    XQUERY = tf.constant(_XQUERY, dtype=tfdt)
    __X1 = ABLR_learner.sess.run(QP1.points[:, 0])
    __Y1 = ABLR_learner.sess.run(QP1.points[:, 1])
    __X2 = ABLR_learner.sess.run(QP2.points[:, 0])
    __Y2 = ABLR_learner.sess.run(QP2.points[:, 1])

    if use_pchip_asymp is True:
        ret1 = ABLR_learner.sess.run(
            bbq_sampler(QP1.points, XQUERY, partial(pchiptx_asymp, tfdt=tfdt)))
        ret2 = ABLR_learner.sess.run(
            bbq_sampler(QP2.points, XQUERY, partial(pchiptx_asymp, tfdt=tfdt)))
    else:
        ret = ABLR_learner.sess.run(
            bbq_sampler(QP1.points, XQUERY, pchiptx_full))

    ax = plt.subplot(3, 3, 3)
    ax.plot(__X1, __Y1, c=CBCOLS["red"], lw=1.1, marker="d", ms=3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_title("Raw interpolation points (before pchip)")
    ax = plt.subplot(3, 3, 6)
    ax.plot(_XQUERY, ret1, c=CBCOLS["blue_medium"], lw=1.2)
    ax.set_xlim(-0.01, 1.01)
    ax.set_title("Interpolated quantile")

    ax = plt.subplot(3, 3, 8)
    ax.plot(__X2, __Y2, c=CBCOLS["red"], lw=1.1, marker="d", ms=3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_title("Raw interpolation points (before pchip)")
    ax = plt.subplot(3, 3, 9)
    ax.plot(_XQUERY, ret2, c=CBCOLS["blue_medium"], lw=1.2)
    ax.set_xlim(-0.01, 1.01)
    ax.set_title("Interpolated quantile")

    plt.tight_layout()
    plt.show()
