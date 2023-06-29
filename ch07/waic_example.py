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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

import stats_rethinking as sts

from pathlib import Path
from scipy import stats

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
    quap = sts.quap(data=df)

Ns = 1000
post = quap.sample(Ns)

# Plot the fit
mu_samp = sts.lmeval(quap, out=quap.model.μ)
ax = sts.lmplot(fit_x=df['speed'], fit_y=mu_samp, data=df, x='speed', y='dist')

#  Log-likelihood of each observation at each sample from the posterior
# (R code 7.21)
loglik = stats.norm(mu_samp, post['σ']).logpdf(df[['dist']])  # (N, Ns)

# NOTE how do we compute the log-likelihood directly from the model, without
# having to rely on stats.norm and doing our own evaluation of the parameters?
# `pm.compile_logp` seems to use the joint probability of the *priors*, with no
# ability to input specific values (i.e. posterior samples) for the value
# variables.

# Take the log of the average over each data point (R code 7.22)
lppd = sts.logsumexp(loglik, axis=1) - np.log(Ns)  # (N,)

# Penalty term (R code 7.23)
p_WAIC = np.var(loglik, axis=1)

# Combine! (R code 7.24)
WAIC = -2 * (lppd.sum() - p_WAIC.sum())
print(f"{WAIC = :.4f}")

# Estimate the standard error of the WAIC (R code 7.25)
waic_vec = -2 * (lppd - p_WAIC)
std_WAIC = (N * np.var(waic_vec))**0.5
print(f"{std_WAIC = :.4f}")

assert np.isclose(WAIC, waic_vec.sum())

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
