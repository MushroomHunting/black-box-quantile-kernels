import numpy as np
from functools import partial
from tqdm import tqdm
import nlopt
import matplotlib.pylab as plt

import bbq.bbq_numpy.models.rff as ff
from bbq.bbq_numpy.examples.utils import toy_functions
from bbq.bbq_numpy.examples.utils.train_model import train_model
from bbq.bbq_numpy.examples.utils.saveResults import ResultSaver
from bbq.bbq_numpy.utils.enums import QMC_KWARG, QMC_SCRAMBLING, QMC_SEQUENCE
from bbq.bbq_numpy.interpolate import InterpWithAsymptotes
from bbq.bbq_numpy.models.ABLR import ABLR
from bbq.bbq_numpy.utils import datasets
from bbq.bbq_numpy import parametrisations
from bbq.bbq_numpy.utils.metrics import mnll

tqdm.monitor_interval = 0

"""
Options
"""
# Dataset
# datasetName = "tread"
# datasetName = "pores"
# datasetName = "rubber"
# datasetName = "concrete"
# datasetName = "airfoil_noise"
# datasetName = "co2"
# datasetName = "airline"
# datasetName = "cosine"
# datasetName = "pattern"
datasetName = "steps"
# datasetName = "quadraticCos"
# datasetName = "harmonics"
# datasetName = "heaviside"

extrapolationDataset = True  # Better plotting for extrapolation dataset
# BLR Model
M = 300  # Number of random fourier features in BLR
blrAlpha, blrBeta = 1, 10  # Precision params for BLR (weight prior, noise)
learnBeta = False  # Learn noise precision
# Quantile type
# useARDkernel = True  # To use an ARD kernel
useARDkernel = False  # To use an isotropic kernel
basicRBF = False  # To run basic RFF with RBF (opt lengthscale)
# quantileParametrisation = parametrisations.InterpPieceWise4points
quantileParametrisation = parametrisations.InterpSingleSpline
# quantileParametrisation = parametrisations.PeriodicSimple
# quantileParametrisation = parametrisations.InterpIncrementY6pts
# quantileParametrisation = parametrisations.BoundedQPoints
# quantileParametrisation = parametrisations.StairCase
# quantileParametrisation = parametrisations.InterpWeibull
# Score metric to use
scoreMetric = "NLML"  # Negative log marginal likelihood
# scoreMetric = "RMSE"  # Root mean square error
# Search strategy
# searchStrategy = "Gridsearch"
# searchStrategy = "BO"  # Bayesian optimisation
searchStrategy = "NLopt"  # Search using NLopt optimisation algorithms
# Grid-search parameters
gridRes = int(np.sqrt(1000))  # Grid-search resolution across each dimension
# BO parameters
nBOIter = 200  # Number of BO iterations
boKappa = 1  # BO Exploration-exploitation parameter
boModelLengthscale = 0.03  # RBF Lenghtscal of approximate GP used in BO
boModelType = "BLR-RFF"  # Approximate GP to be used in BO
# boModelType = "LocalGP"
# NLopt parameters
nlOptIter = 200
# NLopt Global algorithms (derivative free)
# nlOptAlgo = nlopt.GN_CRS
# nlOptAlgo = nlopt.GN_CRS2_LM
# nlOptAlgo = nlopt.GN_DIRECT
# nlOptAlgo = nlopt.GN_DIRECT_L
# nlOptAlgo = nlopt.GN_ISRES
# nlOptAlgo = nlopt.GN_ESCH
# NLopt Local algotithms (derivative free)
nlOptAlgo = nlopt.LN_COBYLA
# nlOptAlgo = nlopt.LN_BOBYQA
# nlOptAlgo = nlopt.LN_SBPLX


"""
Generate dataset
"""
if datasetName == "co2":
    train_data, test_data = datasets.mauna_loa()
elif datasetName == "airline":
    train_data, test_data = datasets.airline_passengers()
elif datasetName == "airfoil_noise":
    train_data, test_data = datasets.airfoil_noise()
elif datasetName == "concrete":
    train_data, test_data = datasets.concrete()
elif datasetName == "rubber" \
        or datasetName == "pores" \
        or datasetName == "tread":
    train_data, test_data = datasets.textures_2D(texture_name=datasetName)
else:
    if datasetName == "cosine":
        objectiveFunction = toy_functions.cosine
    elif datasetName == "harmonics":
        objectiveFunction = toy_functions.harmonics
    elif datasetName == "pattern":
        objectiveFunction = toy_functions.pattern
    elif datasetName == "heaviside":
        objectiveFunction = toy_functions.heaviside
    elif datasetName == "quadraticCos":
        objectiveFunction = toy_functions.quadratic_cos
    elif datasetName == "steps":
        objectiveFunction = toy_functions.steps
    else:
        raise RuntimeError("Objective function was not defined")
    train_x = np.sort(np.random.uniform(-0.3, 1.2, (100, 1)), axis=0)
    train_y = objectiveFunction(train_x)
    train_data = np.hstack([train_x, train_y])
    test_x = np.linspace(-0.5, 1.5, 1000).reshape(-1, 1)
    # test_x = np.sort(np.random.uniform(-1, 1, (100, 1)), axis=0)
    test_y = objectiveFunction(test_x)
    test_data = np.hstack([test_x, test_y])

train_x = train_data[:, :-1]
train_y = train_data[:, [-1]]
test_x = test_data[:, :-1]
test_y = test_data[:, [-1]]
N, D = train_x.shape
print("Dataset size: {} * {}".format(N, D))

if scoreMetric == "RMSE":
    Nv = int(0.2 * N)
    N -= Nv
    val_x = train_x[-Nv:, :]
    val_y = train_y[-Nv:].reshape(-1, 1)
    train_x = train_x[:-Nv, :]
    train_y = train_y[:-Nv].reshape(-1, 1)

vizData = False
if vizData:
    plt.figure()
    plt.plot(train_x, train_y, 'b-', label="train set")
    if scoreMetric == "RMSE":
        plt.plot(val_x, val_y, 'g-', label="validation set")
    plt.plot(test_x, test_y, 'r-', label="test set")
    plt.xlim(min(np.min(train_x), np.min(test_x)) - .2,
             max(np.max(train_x), np.max(test_x)) + .2)
    plt.ylim(min(np.min(train_y), np.min(test_y)) - .2,
             max(np.max(train_y), np.max(test_y)) + .2)
    plt.title("Data set")
    plt.legend(loc=2)
    plt.show()
    exit()

"""
Quantile parametrisation
"""
if quantileParametrisation == parametrisations.BoundedQPoints:
    qp = quantileParametrisation(6, p1=(0.03, -100), p2=(0.97, 100))
elif quantileParametrisation == parametrisations.StairCase:
    qp = quantileParametrisation(9, p1=(0.03, -150), p2=(0.97, 150))
else:
    qp = quantileParametrisation()
genParams = qp.gen_params
interpolator = qp.interpolator
paramLow = qp.paramLow
paramHigh = qp.paramHigh
scaleParams = qp.scale_params
if datasetName == "co2":
    if quantileParametrisation == parametrisations.PeriodicSimple:
        paramHigh[0] = 2.3
        paramLow[0] = 2
        paramHigh[1] = 0.2
        paramLow[1] = 0.01
elif datasetName == "airline":
    if quantileParametrisation == parametrisations.PeriodicSimple:
        paramLow[0] = 2.3
        paramHigh[0] = 2.8
        paramLow[1] = 0.001
elif datasetName == "pores":
    pass
    # paramLow[0] = 1.0
    # paramHigh[0] = 2.8
    # paramLow[1] = 0.01
    # paramHigh[1] = 0.45
elif datasetName == "rubber":
    pass
    # paramLow[0] = 1.0
    # paramHigh[0] = 2.8
    # paramLow[1] = 0.01
    # paramHigh[1] = 0.25
elif datasetName == "tread":
    pass
    # paramLow[0] = 1.0
    # paramHigh[0] = 2.5
    # paramLow[1] = 0.001
    # paramHigh[1] = 0.01

if quantileParametrisation == parametrisations.InterpSingleSpline:
    paramHigh[1] = 6
    paramHigh[0] = 2.3
    paramLow[0] = 1.5

"""
Kernel composition
"""
# composition = ["linear", "+", "linear_no_bias", "*", "bbq"]
# composition = ["linear", "+", "bbq"]
# composition = ["linear", "*", "linear_no_bias", "+", "bbq"]
# composition = ["linear", "+", "linear_no_bias", "*", "bbq"]
composition = ["bbq"]

if datasetName == "co2":
    # composition = ["linear", "*", "linear_no_bias", "+", "bbq"]
    composition = ["linear_no_bias", "+", "bbq"]
elif datasetName == "airline":
    composition = ["linear", "+", "linear", "*", "bbq"]
    # composition = ["linear", "*", "linear_no_bias", "+",
    #                "linear", "*", "bbq"]
    # composition = ["linear", "*", "linear_no_bias", "+",
    #                "linear",  "*", "linear_no_bias", "*", "bbq", "+", "bbq"]

if basicRBF:
    # composition = ["rbf" if e == "bbq" else e for e in composition]
    composition = ["rbf"]
    qp.paramLow = np.array([-3])
    qp.paramHigh = np.array([0])
    qp.logScaleParams = np.array([True])
    paramLow = qp.paramLow
    paramHigh = qp.paramHigh

"""
Search/optimisation for best quantile
"""
if useARDkernel:
    paramLow = np.array(D * list(paramLow))
    paramHigh = np.array(D * list(paramHigh))
    qp.logScaleParams = np.array(D * list(qp.logScaleParams))
if learnBeta:
    paramLow = np.array([-3] + list(paramLow))
    paramHigh = np.array([3] + list(paramHigh))
print("parameter space high: {}".format(paramHigh))
print("parameter space low:  {}".format(paramLow))
score, allParams, bo = None, None, None
if searchStrategy == "Gridsearch":
    assert (len(paramLow) == 2 and "Gridsearch only designed for 2 params")
    X, Y = np.meshgrid(np.linspace(paramLow[0], paramHigh[0], gridRes),
                       np.linspace(paramLow[1], paramHigh[1], gridRes))
    allParams = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])

    score = np.zeros((gridRes ** 2,))
    for i in tqdm(range(allParams.shape[0])):
        if learnBeta:
            blrBeta = 10 ** (allParams[i, 0])
            curParams = allParams[i, 1:]
        else:
            curParams = allParams[i, :]
        scaledParams = scaleParams(curParams)
        score[i] = -train_model(scaledParams, blrBeta)
elif searchStrategy == "BO":
    search_int = np.vstack([np.atleast_2d(paramLow),
                            np.atleast_2d(paramHigh)]).T
    bo = None  # BO.BO(search_int, acq_fun=BO.UCB(boKappa), opt_maxeval=20,
    allParams = np.zeros((nBOIter, len(paramLow)))
    score = np.zeros((nBOIter,))
    for i in tqdm(range(nBOIter)):
        allParams[i, :] = bo.next_sample()
        if learnBeta:
            blrBeta = 10 ** (allParams[i, 0])
            curParams = allParams[i, 1:]
        else:
            curParams = allParams[i, :]
        scaledParams = scaleParams(curParams)
        score[i] = -train_model(scaledParams, blrBeta)
        bo.update(allParams[i, :].reshape(1, -1),
                  np.array(score[i]).reshape(-1, 1))
elif searchStrategy == "NLopt":
    allParams = np.zeros((nlOptIter, len(paramLow)))
    score = np.zeros((nlOptIter,))
    pbar = tqdm(total=nlOptIter)
    i = 0

    # Nlopt params
    opt = nlopt.opt(nlOptAlgo, len(paramLow))
    opt.set_lower_bounds(np.array(paramLow).reshape(-1))
    opt.set_upper_bounds(np.array(paramHigh).reshape(-1))
    opt.set_maxeval(nlOptIter)


    def _fun_maximize(_x, grad):
        global i
        if i == nlOptIter:
            print("Warning: maximum number of iterations reached.")
            return float(np.min(score))
        allParams[i, :] = _x
        if learnBeta:
            global blrBeta
            blrBeta = 10 ** (allParams[i, 0])
            curParams = allParams[i, 1:]
        else:
            curParams = allParams[i, :]

        scaledParams = scaleParams(curParams)
        curScore = float(-train_model(scaledParams, blrBeta))

        # Keep track of previous tries
        score[i] = curScore
        i += 1
        pbar.update(1)

        return curScore


    opt.set_max_objective(_fun_maximize)
    init_opt = np.random.uniform(0, 1, len(paramLow)) * \
               (paramHigh - paramLow) + paramLow
    opt.optimize(init_opt)
    pbar.close()

"""
Saving
"""
# Compute best quantile
imax = np.argmax(score)
curParams = allParams[imax, :]
if learnBeta:
    curParams = allParams[imax, 1:]
    blrBeta = 10 ** allParams[imax, 0]
    print("Best BLR beta: {}".format(blrBeta))
bestParam = scaleParams(curParams)
print("Best parameters: {}".format(bestParam))
bbq_qf = None
if not basicRBF:
    # Interpolation
    if useARDkernel:
        bbq_qf = []
        nprm = len(bestParam)
        for j in range(D):
            x, y, params = genParams(
                bestParam[int(j * nprm / D):int((j + 1) * nprm / D)])
            bbq_qf.append(InterpWithAsymptotes(x=x, y=y,
                                               interpolator=interpolator,
                                               params=params))
    else:
        x, y, params = genParams(bestParam)
        bbq_qf = InterpWithAsymptotes(x=x, y=y, interpolator=interpolator,
                                      params=params)

# Draw features with interpolated quantile
f_dict = {"linear_no_bias": ff.Linear(d=D, has_bias_term=False),
          "linear": ff.Linear(d=D, has_bias_term=True)}

if basicRBF:
    f_dict["rbf"] = ff.RFF_RBF(m=M, d=D, ls=bestParam)
else:
    f_dict["bbq"] = ff.QMCF_BBQ(m=M, d=D,
                                sampler=partial(freqGenFn, bbq_qf=bbq_qf),
                                sequence_type=QMC_SEQUENCE.HALTON,
                                scramble_type=QMC_SCRAMBLING.GENERALISED,
                                qmc_kwargs={QMC_KWARG.PERM: None})

qmcf_bbq = ff.FComposition(f_dict=f_dict, composition=composition)

# Compute best data fit
blr = ABLR(k_phi=qmcf_bbq, d=D, d_out=1,
           alpha=blrAlpha, beta=blrBeta, log_dir=None)
blr.learn_from_history(x_trn=train_x, y_trn=train_y)
prediction = blr.predict(x_tst=train_x, pred_var=True)
pred_train, predVar_train = prediction.mean, prediction.var

# Compute final score metrics
prediction = blr.predict(x_tst=test_x, pred_var=True)
pred_test, predVar_test = prediction.mean, prediction.var
finalRMSE = np.sqrt(np.mean((pred_test - test_y) ** 2))
finalMNLL = mnll(test_y, pred_test, predVar_test)
print("Final model RMSE: {}".format(finalRMSE))
print("Final model MNLL: {}".format(finalMNLL))

settingsVars = {
    "RMSE": finalRMSE,
    "MNLL": finalMNLL,
    "bestParam": bestParam.tolist(),
    "datasetName": datasetName,
    "nlOptIter": nlOptIter,
    "nlOptAlgo": nlOptAlgo,
    "gridRes": gridRes,
    "paramLow": paramLow.tolist(),
    "paramHigh": paramHigh.tolist(),
    "M": M,
    "N": N,
    "D": D,
    "blrAlpha": blrAlpha,
    "blrBeta": blrBeta,
    "useARDkernel": useARDkernel,
    "basicRBF": basicRBF,
    "quantileParametrisation": quantileParametrisation.__name__,
    "scoreMetric": scoreMetric,
    "searchStrategy": searchStrategy,
    "composition": composition,
    "interpolator": interpolator.__name__
}
rs = ResultSaver(settingsVars)
rs.save_params_and_loss(allParams, -score)
if not basicRBF:
    rs.save_quantile(bbq_qf, x, y)
    rs.save_pdf(bbq_qf)

if D != 2:
    if D == 1:
        rs.save_dataset(train_x, train_y, test_x, test_y)
    rs.save_data_fit(train_x, pred_train, predVar_train,
                     test_x, pred_test, predVar_test)
else:
    # Generate images
    imSize = 130
    trainIdx = np.round((train_x + 1) * (imSize - 1) / 2.0).astype(int)
    testIdx = np.round((test_x + 1) * (imSize - 1) / 2.0).astype(int)
    fullImg = np.zeros((imSize, imSize))
    for idx, y in zip(trainIdx, train_y):
        fullImg[idx[0], idx[1]] = y
    for idx, y in zip(testIdx, test_y):
        fullImg[idx[0], idx[1]] = y
    predImg = np.zeros((imSize, imSize))
    for idx, y in zip(trainIdx, pred_train):
        predImg[idx[0], idx[1]] = y
    for idx, y in zip(testIdx, pred_test):
        predImg[idx[0], idx[1]] = y
    predImgPrcs = np.clip(predImg, -1, 1)

    rs.save_pred_image(predImgPrcs)

if len(paramLow) - 1 * learnBeta == 2:  # Plot score map
    if searchStrategy == "BO" or searchStrategy == "NLopt":
        plotRes = 64
        X, Y = np.meshgrid(np.linspace(paramLow[0], paramHigh[0], plotRes),
                           np.linspace(paramLow[1], paramHigh[1], plotRes))
    Z = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
    scaledZ = scaleParams(Z)
    scaledX = scaledZ[:, 0].reshape(X.shape)
    scaledY = scaledZ[:, 1].reshape(Y.shape)
    if searchStrategy == "Gridsearch":
        score_surface = -score
    elif searchStrategy == "BO":
        score_surface = -bo.model.predictMean(Z)
    if searchStrategy == "BO" or searchStrategy == "Gridsearch":
        rs.save_loss_surface(scaledX, scaledY, score_surface.reshape(X.shape))
print("Finished.")
