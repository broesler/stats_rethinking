#!/usr/bin/env python3
# =============================================================================
#     File: api_test.py
#  Created: 2023-07-20 11:24
#   Author: Bernie Roesler
#
"""
Description: Create model with multiple variable types to test API.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats

import stats_rethinking as sts

N = 20
k = 4  # number of parameters

rho = np.r_[0.15, -0.4, 0.5]

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
# === array([[ 1.  ,  0.15, -0.4, 0.5 ],
#            [ 0.15,  1.  ,  0. , 0.  ],
#            [-0.4 ,  0.  ,  1. , 0.  ],
#            [ 0.5 ,  0.  ,  0. , 1.  ]])
#

true_dist = stats.multivariate_normal(mean=np.zeros(n_dim), cov=Rho)

df = (pd.DataFrame(true_dist.rvs(N))  # (N, k)
        .rename({0: 'y'}, axis='columns')
        .set_index('y')
        .add_prefix('X__')
        .reset_index()
        .sort_values('X__1')
      )

# Define the training matrix
mm_train = np.c_[np.ones((N, 1)), df.filter(like='X')]

# Build and fit the model to the training data
with pm.Model() as model:
    # The data
    X = pm.MutableData('X', mm_train)
    obs = pm.MutableData('obs', df['y'])
    # Add a scalar parameter just for testing
    γ = pm.Normal('γ', 0.5, 1)
    # Linear model
    α = pm.Normal('α', 0, 1, shape=(1,))
    βn = pm.Normal('βn', 0, 1, shape=(k-1,))
    β = pm.math.concatenate([α, βn])
    μ = pm.Deterministic('μ', pm.math.dot(X, β))
    y = pm.Normal('y', μ, 1, observed=obs, shape=obs.shape)
    # Fit the model
    q = sts.quap()


sts.precis(q)

# # Plot the fit
# mu_samp = sts.lmeval(q, out=q.model.μ)
# ax = sts.lmplot(fit_x=df['X__1'], fit_y=mu_samp, data=df, x='X__1', y='y')
# plt.ion()
# plt.show()

# =============================================================================
# =============================================================================
