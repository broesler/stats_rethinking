#!/usr/bin/env python3
# =============================================================================
#     File: waic_example.py
#  Created: 2023-06-27 21:43
#   Author: Bernie Roesler
#
"""
Overthinking: WAIC Calculations (R Code 7.20 - 25)
"""
# =============================================================================

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

import stats_rethinking as sts

from pathlib import Path
# from scipy import stats

plt.style.use('seaborn-v0_8-darkgrid')

# (R code 7.20)
data_file = Path('../data/cars.csv')
df = pd.read_csv(data_file, index_col=0).astype(float)
N = len(df)

# Build a linear model of distance ~ speed
with pm.Model():
    a = pm.Normal('a', 0, 100)
    b = pm.Normal('b', 0, 10)
    μ = pm.Deterministic('μ', a + b * df['speed'])
    σ = pm.Exponential('σ', 1)
    dist = pm.Normal('dist', μ, σ, observed=df['dist'])
    q = sts.quap(data=df)

Ns = 1000
post = q.sample(Ns)  # (Ns, Np)

# Plot the fit
mu_samp = sts.lmeval(q, out=q.model.μ)
ax = sts.lmplot(fit_x=df['speed'], fit_y=mu_samp, data=df, x='speed', y='dist')
ax.set_title('Cars')

# Log-likelihood of each observation at each sample from the posterior
# (R code 7.21)
# >>> loglik = stats.norm(mu_samp, post['σ']).logpdf(df[['dist']])  # (N, Ns)
# NOTE Instead, we can use pymc to compute it for us from the model without
# having to compute the mean and reimplement the logpdf ourselves.

# Create InferenceData object with 'posterior' attribute xarray DataSet.
idata = az.convert_to_inference_data(post.expand_dims('chain'))

# Add log_likelihood to idata
idata = pm.compute_log_likelihood(idata, model=q.model, progressbar=False)

# Extract the relevant item
loglik = idata.log_likelihood.mean('chain')  # Dataset: 'dist' (Ns, N)

# quap.loglik ≈ -loglik.mean(axis=1).sum()?

# Take the log of the average over each data point (R code 7.22)
lppd = sts.logsumexp(loglik['dist'], axis=0) - np.log(Ns)  # (Ns, N) -> (N,)

# Penalty term (R code 7.23)
p_WAIC = loglik['dist'].var(dim='draw')  # (Ns, N) -> (N,)

# Combine! (R code 7.24)
WAIC = -2 * (lppd.sum() - p_WAIC.sum())
print(f"{WAIC = :.4f}")  # ~ 423.4012

# Estimate the standard error of the WAIC (R code 7.25)
waic_vec = -2 * (lppd - p_WAIC)
std_WAIC = (N * np.var(waic_vec))**0.5
print(f"{std_WAIC = :.4f}")  # ~ 17.7069

assert np.isclose(WAIC, waic_vec.sum())

# Test WAIC function with log likelihood
wx = sts.WAIC(loglik=loglik)['dist']
print(f"{wx['waic'] = :.4f}")
print(f"{wx['std'] = :.4f}")

# Should be exact
np.testing.assert_allclose(wx['waic'], WAIC)
np.testing.assert_allclose(wx['std'], std_WAIC)

# Test WAIC function with model itself
ww = sts.WAIC(model=q, post=post)['dist']

# Should be exact if we give post
print(f"{ww['waic'] = :.4f}")
print(f"{ww['std'] = :.4f}")

np.testing.assert_allclose(wx['waic'], WAIC)
np.testing.assert_allclose(wx['std'], std_WAIC)

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
