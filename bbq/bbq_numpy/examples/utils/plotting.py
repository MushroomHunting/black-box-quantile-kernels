import numpy as np
import scipy.stats
import matplotlib.pylab as plt

from bbq.bbq_numpy.interpolate import InterpWithAsymptotes


def plot_2d(all_params, all_scores, best_model, qp, X, Y, train_x, train_y):
    """
    Handy function for plotting learned model and dataset when using 2
    parameters.
    :param all_params: all parameters
    :param all_scores: scores for all parameters
    :param best_model: model trained with best parameters
    :param qp: quantile parametrisation
    :param X: first dimension of parameter meshgrid
    :param Y: second dimension of parameter meshgrid
    :param train_x: training dataset inputs
    :param train_y: training dataset targets
    """
    i_max = np.argmax(all_scores)
    cur_params = all_params[i_max, :]
    best_param = qp.scale_params(cur_params)
    print("Best quantile parameters: {}".format(best_param))
    qmcf_bbq = best_model.k_phi.f_dict["bbq"]
    plotLin = np.linspace(-5, 5, 1000)
    phi_bbq = qmcf_bbq.transform(x=plotLin.reshape(-1, 1))
    K_bbq_reconstructed = np.dot(phi_bbq, phi_bbq.T)
    gram_slice = np.hstack((K_bbq_reconstructed[:-1, 0][::-1],
                            K_bbq_reconstructed[0, :]))

    # Compute best data fit
    linFit = np.linspace(-1.0, 2.0, 1000)
    prediction = best_model.predict(linFit.reshape(-1, 1))

    # Plot score map
    plt.figure(figsize=(18, 8))
    plt.subplot(2, 2, 1)
    Z = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
    scaledZ = qp.scale_params(Z)
    scaledX = scaledZ[:, 0].reshape(X.shape)
    scaledY = scaledZ[:, 1].reshape(Y.shape)
    cs = plt.contourf(scaledX, scaledY, -all_scores.reshape(X.shape), 64)
    for c in cs.collections:
        c.set_edgecolor("face")
    plt.plot(best_param[0], best_param[1], "r*")
    plt.colorbar(cs)
    plt.xlabel("param 0")
    plt.ylabel("param 1")
    plt.title("BLR score for two params (red is best argmax)")

    # Plot quantile function
    plt.subplot(2, 2, 2)
    x, y, params = qp.gen_params(best_param)
    bbq_qf = InterpWithAsymptotes(x=x, y=y, interpolator=qp.interpolator,
                                  params=params)
    lin = np.linspace(0, 1, 1000)
    plt.plot(lin, scipy.stats.norm.ppf(lin), 'r', alpha=0.2,
             label='quantile norm')
    plt.plot(lin, bbq_qf(lin), 'b', label='learned quantile')
    plt.plot(x, y, 'ro')
    plt.xlim(0, 1)
    plt.ylim(y[0] - 0.2 * abs(y[0]), y[-1] + 0.2 * abs(y[-1]))
    plt.legend(loc=2)
    plt.title("Best quantile")

    # Plot reconstructed kernel
    plt.subplot(2, 2, 3)
    plt.plot(range(len(gram_slice)), gram_slice)
    plt.xlim(0, len(gram_slice))
    plt.title("Gramm slice")

    # Plot data fit
    plt.subplot(2, 2, 4)
    plt.plot(train_x, train_y, 'r*', label="train data")
    plt.plot(linFit, prediction.mean, 'b-', label="prediction")
    plt.fill_between(linFit.reshape(-1),
                     (prediction.mean - 1.0 * prediction.var).reshape(-1),
                     (prediction.mean + 1.0 * prediction.var).reshape(-1),
                     alpha=0.3, label="predictive variance")
    plt.xlim(linFit[0], linFit[-1])
    plt.ylim(np.min(train_y) - .2, np.max(train_y) + .2)
    plt.title("Data fit with score {}".format(all_scores[i_max]))
    plt.legend(loc=2)
    # plt.savefig('fig.pdf', dpi=300, format='pdf')
    plt.show()
