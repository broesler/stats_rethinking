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
# import xarray as xr

from scipy import stats

import stats_rethinking as sts

Y_SIGMA = 1
N = 20
k = 3
rho = np.r_[0.15, -0.4]
b_sigma = 0.5


def loglik(m, data_in, data_out, Ns=1000):
    """Compute the log-likelihood of the data, given the model."""
    mu_samp = sts.lmeval(m, out=m.model.μ, eval_at={'X': data_in}, N=Ns)
    return stats.norm(mu_samp, Y_SIGMA).logpdf(np.c_[data_out])


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
with pm.Model():
    X = pm.MutableData('X', mm_train)
    if k == 1:
        α = pm.Normal('α', 0, b_sigma)
        μ = pm.Deterministic('μ', α)
        y = pm.Normal('y', μ, Y_SIGMA, observed=y_train)
    else:
        α = pm.Normal('α', 0, b_sigma, shape=(1,))
        βn = pm.Normal('βn', 0, b_sigma, shape=(k-1,))
        β = pm.math.concatenate([α, βn])
        μ = pm.Deterministic('μ', pm.math.dot(X, β))
        y = pm.Normal('y', μ, Y_SIGMA,
                      observed=y_train,
                      shape=X[:, 0].shape
                      )
    q = sts.quap()

# Compute the deviance
dev = pd.Series(index=['train', 'test'], dtype=float)
dev['train'] = -2 * np.sum(lppd(q, data_in=mm_train, data_out=y_train))

# Compute the lppd with the test data
mm_test = np.ones((N, 1))
if k > 1:
    mm_test = np.c_[mm_test, X_test[:, :k-1]]

dev['test'] = -2 * np.sum(lppd(q, data_in=mm_test, data_out=y_test))

# Compute WAIC, LOOIC, and LOOCV
wx = WAIC(q, data_in=mm_test, data_out=y_test)
waic_s = pd.Series({'test': wx['waic'],
                    'err': np.abs(wx['waic'] - dev['test'])})

# -----------------------------------------------------------------------------
#         Test LOOIC computation
# -----------------------------------------------------------------------------
Ns = 1000
y_loglik = loglik(q, mm_test, y_test, Ns)              # (N, Ns)
y_lppd = sts.logsumexp(y_loglik, axis=1) - np.log(Ns)  # (N,)

post = q.sample(1000)  # DataFrame with columns = ['α', 'βn__0', 'βn__1', ...]


def frame_to_array(df):
    """Convert DataFrame to ArviZ DataSet by combinining columns with
    multi-dimensional parameters, e.g. β__0, β__1, ..., β__N into β (N,).
    """
    tf = df.copy()
    var_names = tf.columns.str.replace('__[0-9]+', '', regex=True).unique()
    the_dict = dict()
    for v in var_names:
        the_dict[v] = np.expand_dims(tf.filter(like=v).values, 0)
    return az.convert_to_dataset(the_dict)


def array_to_frame(ds):
    """Convert ArviZ DataSet to DataFrame by separating columns with
    multi-dimensional parameters, e.g. β (N,) into β__0, β__1, ..., β__N.
    """
    df = pd.DataFrame()
    for v_name, var in ds.data_vars.items():
        v = var.mean('chain').squeeze()  # remove chain dimension
        if v.ndim == 1:                  # only draw dimension
            df[v_name] = v.values
        elif v.ndim > 1:
            df[_names_from_vec(v_name, v.shape[1])] = v.values
        else:
            raise ValueError(f"{v_name} has invalid dimension {v.ndim}.")
    return df


def _names_from_vec(v_name, ncols):
    """Create a list of strings ['x__0', 'x__1', ..., 'x__``ncols``'],
    where 'x' is ``v_name``."""
    # TODO case of 2D, etc. variables
    fmt = '02d' if ncols > 10 else 'd'
    return [f"{v_name}__{i:{fmt}}" for i in range(ncols)]


da = frame_to_array(post)
df = array_to_frame(da)
idata = az.convert_to_inference_data(da)
idata = pm.compute_log_likelihood(idata, model=q.model, progressbar=False)

loo = az.loo(idata)

res = pd.concat([dev, waic_s], keys=['deviance', 'WAIC'])

# -----------------------------------------------------------------------------
#         Look at actual pymc/arviz examples
# -----------------------------------------------------------------------------
trace = pm.sample(model=q.model, idata_kwargs={'log_likelihood': True})
data = az.load_arviz_data('centered_eight')

# =============================================================================
# =============================================================================
