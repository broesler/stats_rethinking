#!/usr/bin/env python3
# =============================================================================
#     File: pymc_ic_test.py
#  Created: 2023-06-28 13:51
#   Author: Bernie Roesler
#
"""
Tests arviz information criteria.
"""
# =============================================================================

import arviz as az
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats
# from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import stats_rethinking as sts

# Function arguments
N = 20
k = 3
rho = np.r_[0.15, -0.4]
b_sigma = 0.5

# Internal model parameters
Y_SIGMA = 1


def frame_to_dataset(df):
    """Convert DataFrame to ArviZ DataSet by combinining columns with
    multi-dimensional parameters, e.g. β__0, β__1, ..., β__N into β (N,).
    """
    tf = df.copy()
    var_names = tf.columns.str.replace('__[0-9]+', '', regex=True).unique()
    the_dict = dict()
    for v in var_names:
        the_dict[v] = np.expand_dims(tf.filter(like=v).values, 0)
    return az.convert_to_dataset(the_dict)


def dataset_to_frame(ds):
    """Convert ArviZ DataSet to DataFrame by separating columns with
    multi-dimensional parameters, e.g. β (N,) into β__0, β__1, ..., β__N.
    """
    df = pd.DataFrame()
    for vname, var in ds.data_vars.items():
        v = var.mean('chain').squeeze()  # remove chain dimension
        if v.ndim == 1:                  # only draw dimension
            df[vname] = v.values
        elif v.ndim > 1:
            df[_names_from_vec(vname, v.shape[1])] = v.values
        else:
            raise ValueError(f"{vname} has invalid dimension {v.ndim}.")
    return df


def _names_from_vec(vname, ncols):
    """Create a list of strings ['x__0', 'x__1', ..., 'x__``ncols``'],
    where 'x' is ``vname``."""
    # TODO case of 2D, etc. variables
    fmt = '02d' if ncols > 10 else 'd'
    return [f"{vname}__{i:{fmt}}" for i in range(ncols)]


# TODO refactor these functions into generics

def loglik(quap, var_names=None, eval_at=None, Ns=1000):
    """Compute the log-likelihood of the data, given the model.

    Parameters
    ----------
    quap : :obj:`Quap`
        The fitted model object.
    var_names : sequence of str
        List of observed variables for which to compute log likelihood.
        Defaults to all observed variables.
    eval_at : dict like {var_name: values}
        The data over which to evaluate the log likelihood. If not given, the
        data currently in the model is used.
    Ns : int
        The number of samples to take of the posterior.

    Returns
    -------
    result : xarray.Dataset
        Log likelihood for each of the ``var_names``. Each will be an array of
        size (Ns, N), for ``Ns`` samples of the posterior, and `N` data points.
    """
    post = quap.sample(Ns)  # DataFrame with ['α', 'βn__0', 'βn__1', ...]

    if eval_at is not None:
        for k, v in eval_at.items():
            quap.model.set_data(k, v)

    idata = pm.compute_log_likelihood(
        idata=az.convert_to_inference_data(frame_to_dataset(post)),
        model=quap.model,
        var_names=var_names,
        progressbar=False,
    )
    return idata.log_likelihood.mean('chain')


def lppd(m, data_in, data_out, Ns=1000):
    """Compute the log pointwise predictive density for a model."""
    return sts.logsumexp(loglik(m, data_in, data_out, Ns), axis=1) - np.log(Ns)


def WAIC(m, data_in, data_out, Ns=1000, pointwise=False):
    """Compute the Widely Applicable Information Criteria for the model."""
    y_loglik = loglik(m, data_in, data_out, Ns=1000)
    y_lppd = sts.logsumexp(y_loglik, axis=1) - np.log(Ns)
    penalty = np.var(y_loglik, axis=1)
    waic_vec = -2 * (y_lppd - penalty)
    n_cases = y_loglik.shape[0]
    std_err = (n_cases * np.var(waic_vec))**0.5
    w = waic_vec if pointwise else waic_vec.sum()
    return dict(waic=w, lppd=y_lppd, penalty=penalty, std=std_err)


def LOO(m, data_in, data_out, Ns=1000, pointwise=False):
    """Compute the Pareto-smoothed Importance Sampling Leave-One-Out
    Cross-Validation score of the model."""
    # y_loglik = logp(m, data_in, data_out, Ns=1000)
    # y_lppd = logsumexp(y_loglik, axis=1) - np.log(Ns)
    # loo = az.loo(...)
    pass


# Define the dimensions of the "true" distribution to match the model
n_dim = 1 + len(rho)
if n_dim < k:
    n_dim = k

# Generate train/test data
# NOTE this method of creating the data obfuscates the underlying linear
# model, but it is a clean way of accommodating varying parameter lengths.
Rho = np.eye(n_dim)
Rho[0, 1:len(rho)+1] = rho
Rho[1:len(rho)+1, 0] = rho

# >>> Rho
# === array([[ 1.  ,  0.15, -0.4 ],
#            [ 0.15,  1.  ,  0.  ],
#            [-0.4 ,  0.  ,  1.  ]])
#

true_dist = stats.multivariate_normal(mean=np.zeros(n_dim), cov=Rho)
X_train = true_dist.rvs(N)  # (N, k)
X_test = true_dist.rvs(N)

# Separate the inputs and outputs for readability
y_train, X_train = X_train[:, 0], X_train[:, 1:]
y_test, X_test = X_test[:, 0], X_test[:, 1:]

# Define the training matrix
mm_train = np.ones((N, 1))  # intercept term
if k > 1:
    mm_train = np.c_[mm_train, X_train[:, :k-1]]

# Build and fit the model to the training data
# NOTE In order to use `pm.compute_log_likelihood` we need to have ind/obs
# variables in the model itself, and set them to be mm_test/y_test.
with pm.Model():
    X = pm.MutableData('X', mm_train)
    obs = pm.MutableData('obs', y_train)
    if k == 1:
        α = pm.Normal('α', 0, b_sigma)
        μ = pm.Deterministic('μ', α)
        y = pm.Normal('y', μ, Y_SIGMA, observed=obs)
    else:
        α = pm.Normal('α', 0, b_sigma, shape=(1,))
        βn = pm.Normal('βn', 0, b_sigma, shape=(k-1,))
        β = pm.math.concatenate([α, βn])
        μ = pm.Deterministic('μ', pm.math.dot(X, β))
        y = pm.Normal('y', μ, Y_SIGMA, observed=obs, shape=obs.shape)
    q = sts.quap()

# Convert posterior to az.InferenceData for use in pymc log_likelihood
# computation.
# NOTE see pymc.stats.compute_log_likelihood source:
# elemwise_loglike_fn = model.compile_fn(
#            inputs=model.value_vars,
#            outs=model.logp(vars=observed_vars, sum=False),
#            on_unused_input="ignore",
#        )

# Compute the log likelihood
Ns = 1000

# TODO move to testing script for frame_to_dataset/dataset_to_frame
# post = q.sample(Ns)  # DataFrame with columns = ['α', 'βn__0', 'βn__1', ...]
# da = frame_to_dataset(post)
# post = dataset_to_frame(da)  # test inverse function

# Compute the deviance
dev = pd.Series(index=['train', 'test'], dtype=float)

loglik_train = loglik(q)['y']  # (Ns, N)
lppd_train = sts.logsumexp(loglik_train, axis=0) - np.log(Ns)
dev['train'] = -2 * np.sum(lppd_train)

# Compute the lppd with the test data
mm_test = np.ones((N, 1))
if k > 1:
    mm_test = np.c_[mm_test, X_test[:, :k-1]]

loglik_test = loglik(q, eval_at={'X': mm_test, 'obs': y_test})['y']
lppd_test = sts.logsumexp(loglik_test, axis=0) - np.log(Ns)
dev['test'] = -2 * np.sum(lppd_test)

# NOTE how to convert these into functions and not have to recompute the
# log-likelihood each time?
#
# Solution: multiple dispatch:
#
# rethinking::WAIC and rethinking::LOO/PSIS take:
#   * quap and re-computes log-likelihood, lppd, and penalty term
#   * list[['log-likelihood',]] and computes lppd and penalty term
#
# rethinking::cv_quap takes quap_model only
#
# rethinking::LOO/PSIS uses outside library, others computes internally


# -----------------------------------------------------------------------------
#         Compute WAIC
# -----------------------------------------------------------------------------
# wx = WAIC(q, data_in=mm_test, data_out=y_test)
pointwise = False
penalty = np.var(loglik_test, axis=0)
waic_vec = -2 * (lppd_test - penalty)
n_cases = loglik_test.shape[1]
std_err = (n_cases * np.var(waic_vec))**0.5
lppd_w = lppd_test if pointwise else lppd_test.sum()
w = waic_vec if pointwise else waic_vec.sum()
p = penalty if pointwise else penalty.sum()
wx = dict(waic=w, lppd=lppd_w, penalty=p, std=std_err)

waic_s = pd.Series({'test': wx['waic'],
                    'err': np.abs(wx['waic'] - dev['test'])})

# -----------------------------------------------------------------------------
#         Compute LOOIC
# -----------------------------------------------------------------------------
# with warnings.catch_warnings():
#     warnings.simplefilter('ignore', category=UserWarning)

loo = az.loo(idata_test, pointwise=pointwise)

# print(loo)

lx = dict(
    PSIS=-2*loo.elpd_loo,  # == loo_list$estimates['looic', 'Estimate']
    lppd=loo.elpd_loo,
    penalty=loo.p_loo,     # == loo_list$p_loo
    std=2*loo.se,          # == loo_list$estimates['looic', 'SE']
)

psis_s = pd.Series({'test': lx['PSIS'],
                    'err': np.abs(lx['PSIS'] - dev['test'])})

# -----------------------------------------------------------------------------
#         Compute LOOCV
# -----------------------------------------------------------------------------
# See rethinking::cv_quap -> expensive!!
# outcome = q.model.observed_RVs  # user-defined?
N = len(y_train)
# lno = N // 4  # number of data points to leave out per iteration
lno = 1  # number of data points to leave out per iteration
M = N // lno  # number of chunks of data

y_list = np.split(y_train, M)
mm_list = np.split(mm_train, M)


def leave_out(data_list, i):
    """Return an array with one element of the list removed."""
    return np.concatenate([data_list[j] for j in range(M) if j != i])


# Fit a model to each chunk of data
def loocv_func(i):
    """Perform cross-validation on data chunk `i`."""
    # Train the model on the data without chunk i
    q.model.set_data('X', leave_out(mm_list, i))
    q.model.set_data('obs', leave_out(y_list, i))
    the_quap = sts.quap(model=q.model)

    # Compute the LPPD on the left-out chunk of data
    the_quap.model.set_data('X', mm_list[i])
    the_quap.model.set_data('obs', y_list[i])
    post = the_quap.sample(Ns)
    idata = pm.compute_log_likelihood(
        idata=az.convert_to_inference_data(frame_to_dataset(post)),
        model=the_quap.model,
        progressbar=False,
    )
    the_loglik = idata.log_likelihood['y'].mean('chain').values
    the_lppd = sts.logsumexp(the_loglik, axis=0) - np.log(Ns)
    return the_lppd


# NOTE **WARNING** SLOW CODE
lppd_list = process_map(loocv_func, range(M), max_workers=16, desc='LOOCV')

lppd_cv = np.array(lppd_list).squeeze()
c = lppd_cv if pointwise else lppd_cv.sum()

# mean of chunks, var of data
var = lppd_cv.var() if lno == 1 else lppd_cv.mean(axis=0).var()
std_err = (N * var)**0.5

cx = dict(
    loocv=-2*c,
    lppd=c,
    std=std_err,
)

loocv_s = pd.Series({'test': cx['loocv'],
                     'err': np.abs(cx['loocv'] - dev['test'])})

# -----------------------------------------------------------------------------
#         Compile Results
# -----------------------------------------------------------------------------
# res = pd.concat([dev, waic_s, psis_s], keys=['deviance', 'WAIC', 'LOOIC'])
res = pd.concat([dev, waic_s, psis_s, loocv_s],
                keys=['deviance', 'WAIC', 'LOOIC', 'LOOCV'])
print(res)

# -----------------------------------------------------------------------------
#         Look at actual pymc/arviz examples
# -----------------------------------------------------------------------------
# tr = pm.sample(model=q.model, idata_kwargs={'log_likelihood': True})
# loo = az.loo(trace)
# data = az.load_arviz_data('centered_eight')

# =============================================================================
# =============================================================================
