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
import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats

import stats_rethinking as sts

# Function arguments
N = 20
k = 3
rho = np.r_[0.15, -0.4]
b_sigma = 0.5

# Internal model parameters
Y_SIGMA = 1

# -----------------------------------------------------------------------------
#         Define the model
# -----------------------------------------------------------------------------
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
with pm.Model() as model:
    X = pm.MutableData('X', mm_train)
    obs = pm.MutableData('obs', y_train)
    α = pm.Normal('α', 0, b_sigma, shape=(1,))
    if k == 1:
        μ = pm.Deterministic('μ', α)
    else:
        βn = pm.Normal('βn', 0, b_sigma, shape=(k-1,))
        β = pm.math.concatenate([α, βn])
        μ = pm.Deterministic('μ', pm.math.dot(X, β))
    y = pm.Normal('y', μ, Y_SIGMA, observed=obs, shape=obs.shape)
    q = sts.quap()

# -----------------------------------------------------------------------------
#         Compute the Information Criteria
# -----------------------------------------------------------------------------
# NOTE for more efficient computation, we could get the inference data once to
# use for LOOIS, and then extract the log-likelihood for lppd and WAIC.
# LOOIS.
Ns = 1000

post = q.sample(Ns)

da = sts.dataset_to_frame(post)  # test inverse function

# NOTE fix for k == 1 involves setting α shape=(1,) above, *or* not having
# a defined dimension 'α_dim_0' in the Dataset `ds`. Would be better to fix the
# internal frame_to_dataset() function so that we don't have to remember to
# define the model shape each time?
# See: `np.expand_dims` line for all variables, vs. just vectors?
#
# `trace` always works with pm.compute_log_likelihood because pm.sample knows
# the variable shapes! Our method does not have a way to differentiate between
# a singleton variable with shape (), and a vector variable with shape (1,),
# because there is just a single name in the posterior DataFrame either way. If
# we switch the quap() function to use an xarray, we can track the shapes. Then
# the only issue is when we call model.set_data() with new data that has
# varying shape? That shouldn't affect the parameter shapes in the posterior
# though.
#
# `idata.posterior` with α.shape = () should have dims (chain=1, draw=Ns), as
# opposed to dims (chain=1, draw=Ns, α_dim_0=1).

# ds = sts.frame_to_dataset(post, model)
idata = az.convert_to_inference_data(post)

trace = pm.sample(model=q.model)

loglik_train = pm.compute_log_likelihood(
    idata=idata,
    model=q.model,
    progressbar=False,
)

# Compute the lppd
lppd_train = sts.lppd(q)['y']

# Compute the lppd with the test data
mm_test = np.ones((N, 1))
if k > 1:
    mm_test = np.c_[mm_test, X_test[:, :k-1]]

# Compute the posterior and log-likelihood
idata = sts.inference_data(q, eval_at={'X': mm_test, 'obs': y_test})
loglik = idata.log_likelihood.mean('chain')

lppd_test = sts.lppd(loglik=loglik)['y']

# Compute the deviance
res = pd.Series({('deviance', 'train'): -2 * np.sum(lppd_train),
                 ('deviance', 'test'): -2 * np.sum(lppd_test)})

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

# Compile Results
wx = sts.WAIC(loglik=loglik)['y']
res[('WAIC', 'test')] = wx['waic']
res[('WAIC', 'err')] = np.abs(wx['waic'] - res[('deviance', 'test')])

lx = sts.LOOIS(idata=idata)
res[('LOOIC', 'test')] = lx['PSIS']
res[('LOOIC', 'err')] = np.abs(lx['PSIS'] - res[('deviance', 'test')])

cx = sts.LOOCV(
    model=q,
    ind_var='X',
    obs_var='obs',
    out_var='y',
    X_data=mm_train,
    y_data=y_train,
)
res[('LOOCV', 'test')] = cx['loocv']
res[('LOOCV', 'err')] = np.abs(cx['loocv'] - res[('deviance', 'test')])

print(res)

# =============================================================================
# =============================================================================
