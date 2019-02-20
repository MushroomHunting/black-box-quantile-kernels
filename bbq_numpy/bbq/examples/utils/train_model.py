import numpy as np
from functools import partial
import bbq.models.rff as ff
from bbq.utils.enums import QMC_KWARG, QMC_SCRAMBLING, QMC_SEQUENCE
from bbq.interpolate import InterpWithAsymptotes
from bbq.models.ABLR import ABLR
from bbq.utils.math import neg_log_marginal_likelihood as nlml


def QMCF_BBQ(sequence, bbq_qf):
    s = bbq_qf(sequence.points)
    return s.T


def QMCF_BBQ_ARD(sequence, bbq_qf):
    if isinstance(bbq_qf, list):
        d = sequence.points.shape[1]
        s = np.concatenate(
            [bbq_qf[d](sequence.points[:, [d]]) for d in range(d)],
            axis=1)
    else:
        s = bbq_qf(sequence.points)
    return s.T


def train_model(parameters, d, m, blr_alpha, blr_beta, train_x, train_y,
                  val_x, val_y, interpolator, composition, gen_params_function,
                  use_adr_kernel, use_basic_rbf, score_metric):
    """
    Trains BLR on given dataset with specified parameters and returns a score
    """
    bbq_qf = None
    if not use_basic_rbf:
        # Interpolation
        if use_adr_kernel:
            bbq_qf = []
            nprm = len(parameters)
            for j in range(d):
                x, y, params = gen_params_function(
                    parameters[int(j * nprm / d):int((j + 1) * nprm / d)])
                bbq_qf.append(InterpWithAsymptotes(x=x, y=y,
                                                   interpolator=interpolator,
                                                   params=params))
        else:
            x, y, params = gen_params_function(parameters)
            bbq_qf = InterpWithAsymptotes(x=x, y=y, interpolator=interpolator,
                                          params=params)

    # Draw features with interpolated quantile
    f_dict = {"linear_no_bias": ff.Linear(d=d, has_bias_term=False),
              "linear": ff.Linear(d=d, has_bias_term=True)}

    if use_basic_rbf:
        f_dict["rbf"] = ff.RFF_RBF(m=m, d=d, ls=parameters)
    else:
        if use_adr_kernel:
            frequency_generation_function = QMCF_BBQ_ARD
        else:
            frequency_generation_function = QMCF_BBQ
        f_dict["bbq"] = ff.QMCF_BBQ(
            m=m, d=d,
            sampler=partial(frequency_generation_function, bbq_qf=bbq_qf),
            sequence_type=QMC_SEQUENCE.HALTON,
            scramble_type=QMC_SCRAMBLING.GENERALISED,
            qmc_kwargs={QMC_KWARG.PERM: None})

    qmcf_bbq = ff.FComposition(f_dict=f_dict, composition=composition)

    # Run BLR on dataset
    blr = ABLR(k_phi=qmcf_bbq, d=d, d_out=1,
               alpha=blr_alpha, beta=blr_beta, log_dir=None, verbose=0)

    # Learn first on a historical dataset
    blr.learn_from_history(x_trn=train_x, y_trn=train_y, batch_size=100)

    # Compute score
    if score_metric == "RMSE":
        pred = blr.predict(x_tst=val_x, pred_var=False).mean
        return np.sqrt(np.mean((pred - val_y) ** 2))
    elif score_metric == "NLML":
        scr = nlml(y_trn=train_y, phi=qmcf_bbq.transform(train_x),
                   alpha=blr_alpha, beta=blr_beta, s=blr.s, s_inv=blr.s_inv)
        return scr, blr
