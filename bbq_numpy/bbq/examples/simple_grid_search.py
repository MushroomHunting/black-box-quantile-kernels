import numpy as np
from tqdm import tqdm

from bbq import parametrisations
from bbq.examples.utils import toy_functions
from bbq.examples.utils.plotting import plot_2d
from bbq.examples.utils.train_model import train_model

"""
Basic example of optimising a BBQ kernel with grid search. The quantile 
parametrisation uses a simple spline with 2 parameters.
"""

# BLR Model parameters
m = 300  # Number of random fourier features in BLR
blr_alpha, blr_beta = 1, 10  # Precision params for BLR (weight prior, noise)
learn_beta = False  # Learn noise precision

# BBQ parameters
qp = parametrisations.InterpSingleSpline()  # Quantile parametrization (spline)
qp.paramHigh[1] = 3
qp.paramHigh[0] = 3
qp.paramLow[0] = 1.
qp.paramLow[1] = 1.
composition = ["bbq"]  # Kernel composition
score_metric = "NLML"  # Negative log marginal likelihood to train BBQ
gridRes = int(np.sqrt(300))  # Grid-search resolution across each dimension

# Dataset
objectiveFunction = toy_functions.steps
train_x = np.sort(np.random.uniform(-0.3, 1.2, (100, 1)), axis=0)
train_y = objectiveFunction(train_x)
train_data = np.hstack([train_x, train_y])
test_x = np.linspace(-0.5, 1.5, 1000).reshape(-1, 1)
test_y = objectiveFunction(test_x)
test_data = np.hstack([test_x, test_y])
train_x = train_data[:, :-1]
train_y = train_data[:, [-1]]
test_x = test_data[:, :-1]
test_y = test_data[:, [-1]]
n, d = train_x.shape
print("Dataset size: {} * {}".format(n, d))

"""
Start search
"""
X, Y = np.meshgrid(np.linspace(qp.paramLow[0], qp.paramHigh[0], gridRes),
                   np.linspace(qp.paramLow[1], qp.paramHigh[1], gridRes))
all_params = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])

all_scores = np.zeros((gridRes ** 2,))
for i in tqdm(range(all_params.shape[0])):
    cur_params = all_params[i, :]
    scaled_params = qp.scale_params(cur_params)
    all_scores[i] = -train_model(
        scaled_params, d, m, blr_alpha, blr_beta, train_x, train_y,
        None, None, qp.interpolator, composition, qp.gen_params,
        False, False, score_metric)[0]

"""
Plotting
"""
i_max = np.argmax(all_scores)
best_param = qp.scale_params(all_params[i_max, :])
best_model = train_model(best_param, d, m, blr_alpha, blr_beta, train_x,
                         train_y, None, None, qp.interpolator, composition,
                         qp.gen_params, False, False, score_metric)[1]
plot_2d(all_params, all_scores, best_model, qp, X, Y, train_x, train_y)
