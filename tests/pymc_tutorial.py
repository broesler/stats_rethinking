#!/usr/bin/env python3
# =============================================================================
#     File: pymc_tutorial.py
#  Created: 2019-06-20 11:25
#   Author: Bernie Roesler
#
"""
  Description: A script to execute the code in:
               <https://docs.pymc.io/notebooks/getting_started.html>
"""
# =============================================================================

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from matplotlib import cm

plt.style.use('seaborn-v0_8-darkgrid')
rng = np.random.default_rng(123)  # initialize random number generator

N = 100  # size of dataset

# True parameter values
sigma = 1
beta = np.array([[1, 1, 2.5]]).T  # (p+1,1) include "alpha" intercept in beta

# Predictor variable (N, p+1)
X = np.hstack([np.ones((N, 1)),  # add column of ones for intercept
               rng.standard_normal((N, 1)),
               rng.standard_normal((N, 1)) * 0.2])

p = X.shape[1] - 1  # number of features

# Simulate outcome variable (model values + additional noise)
Y = X @ beta + rng.standard_normal((N, 1))*sigma

df = pd.DataFrame(data=np.hstack([X, Y]), columns=['X0', 'X1', 'X2', 'Y'])

# Plot data vs each predictor axis
fig, ax = plt.subplots(2, 1, num=1, sharex=True, sharey=True,
                       figsize=(5, 8), clear=True, constrained_layout=True)
for i in range(len(ax)):
    ax[i].scatter(X[:, i+1], Y)
    ax[i].set_xlabel(f'$X_{i+1}$')
    ax[i].set_ylabel('$Y$')

# Plot bivariate data
fig = plt.figure(2, clear=True, constrained_layout=True)
fig.set_size_inches((6, 3), forward=True)
ax = fig.add_subplot()
cax = ax.scatter(X[:, 1], X[:, 2], c=Y.squeeze(), cmap=cm.inferno)
ax.set_aspect('equal')

ax.set_xlabel('$X_1$')
ax.set_ylabel('$X_2$')

cbar = fig.colorbar(cax, orientation='horizontal', pad=0.1)
cbar.set_label('$Y$')

# -----------------------------------------------------------------------------
#        Create the model
# -----------------------------------------------------------------------------
basic_model = pm.Model()

with basic_model:
    # Priors for unknown model parameters
    beta_h = pm.Normal('beta_h', mu=0, sigma=10, shape=(p+1, 1))
    sigma_h = pm.HalfNormal('sigma_h', sigma=1)

    # Expected value of outcome
    mu = pm.math.dot(X, beta_h)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma_h, observed=Y)

    # Find the MAP estimate of parameter values
    pm.sample()  # initialize NUTS
    map_estimate = pm.find_MAP(model=basic_model)

print("True Params:\n-------------\n"
      f"beta_h = {beta.squeeze()}\n"
      f"sigma_h = {sigma}\n\n"
      "MAP Estimate:\n-------------\n"
      f"beta_h = {map_estimate['beta_h'].squeeze()}\n"
      f"sigma_h = {map_estimate['sigma_h']}")

# Use MCMC estimation
with basic_model:
    step = pm.Slice()  # instantiate sampler
    trace = pm.sample(5000, step=step)  # draw posterior samples

az.plot_trace(trace)
pm.summary(trace).round(2)

plt.show()

# =============================================================================
# =============================================================================
