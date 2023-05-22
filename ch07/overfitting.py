#!/usr/bin/env python3
# =============================================================================
#     File: overfitting.py
#  Created: 2023-05-17 22:42
#   Author: Bernie Roesler
#
"""
§7.1 The Problem with Parameters.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

# Manually define data (R code 7.1)
sppnames = ['afarensis', 'africanus', 'habilis', 'boisei', 'rudolfensis',
            'ergaster', 'sapiens']
brainvolcc = [438, 452, 612, 521, 752, 871, 1350]
masskg = [37.0, 35.5, 34.5, 41.5, 55.5, 61.0, 53.5]

df = pd.DataFrame(dict(species=sppnames, brain=brainvolcc, mass=masskg))

# Standardize variables (R code 7.2)
df['mass_std'] = sts.standardize(df['mass'])
df['brain_std'] = df['brain'] / df['brain'].max()  # [0, 1]

# Use vague priors (R code 7.3)
with pm.Model():
    α = pm.Normal('α', 0.5, 1)
    β = pm.Normal('β', 0, 10)
    μ = pm.Deterministic('μ', α + β * df['mass_std'])
    log_σ = pm.Normal('log_σ', 0, 1)
    brain_std = pm.Normal('brain_std', μ, pm.math.exp(log_σ),
                          observed=df['brain_std'])
    m7_1 = sts.quap(data=df)

print('m7.1:')
sts.precis(m7_1)

# Compute R² value (R code 7.4)
post = m7_1.sample()
mu_samp = sts.lmeval(m7_1, out=m7_1.model.μ, dist=post)
h_samp = stats.norm(mu_samp, np.exp(post['log_σ'])).rvs()
h_mean = h_samp.mean(axis=1)
r = h_samp.mean(axis=1) - df['brain_std']  # residuals
Rsq = 1 - r.var(ddof=0) / df['brain_std'].var(ddof=0)
print(f"{Rsq = :.4f}")

# NOTE Why is this method incorrect? If we can figure this out, we don't need
# to write the sim function to correspond to link! It is just the lmeval with
# a different output variable.
# ==> I believe this method gives an incorrect answer because it does not touch
# the internal rng state of pytensor for each draw, so the draws are *not*
# independent. Thus, the result is a bunch of random points within the PI of
# brain_std, but whose *means* are not necessarily near the brain_std mean.
#
# h_samp = sts.lmeval(m7_1,
#                     out=m7_1.model.brain_std,
#                     params=[m7_1.model.α,
#                             m7_1.model.β,
#                             m7_1.model.log_σ]
#                     )
#


# Plot the data and linear predictions
fig = plt.figure(1, clear=True, constrained_layout=True)
ax = fig.add_subplot()
ax.scatter('mass', 'brain', data=df)
for label, x, y in zip(df['species'], df['mass'], df['brain']):
    ax.text(x+0.5, y+2, label)
ax.scatter(df['mass'], h_mean * df['brain'].max(), c='k', marker='x')
ax.set(xlabel='body mass [kg]',
       ylabel='brain volume [cc]')


# -----------------------------------------------------------------------------
#         Create polynomial models (R code 7.6 - 7.8)
# -----------------------------------------------------------------------------
models = dict()
Rsqs = dict()
Np = 3  # max polynomial terms


# (R code 7.5)
def brain_Rsq(quap):
    """Compute the :math:`R^2` value of the model."""
    post = quap.sample()
    # TODO re-concatenate post columns for multidimensional tensors in lmeval
    mu_samp = sts.lmeval(quap, out=quap.model.μ, dist=post,
                         params=[quap.model.α, quap.model.βn])
    h_samp = stats.norm(mu_samp, np.exp(post['log_σ'])).rvs()
    r = h_samp.mean(axis=1) - df['brain_std']  # residuals
    # pandas default is ddof=1 => N-1, so explicitly use ddof=0 => N.
    return 1 - r.var(ddof=0) / df['brain_std'].var(ddof=0)


for poly_order in range(1, Np+1):
    with pm.Model():
        α = pm.Normal('α', 0.5, 1, shape=(1,))
        βn = pm.Normal('βn', 0, 10, shape=(poly_order,))
        X = sts.design_matrix(df['brain_std'], poly_order)  # [1, x, x², ...]
        β = pm.math.concatenate([α, βn])
        μ = pm.Deterministic('μ', pm.math.dot(X, β))
        log_σ = pm.Normal('log_σ', 0, 1)
        sigma = pm.math.exp(log_σ) if poly_order < 6 else 0.001
        brain_std = pm.Normal('brain_std', μ, sigma, observed=df['brain_std'])
        quap = sts.quap(data=df)
        k = f"m7.{poly_order}"
        models[k] = quap
        print(k)
        sts.precis(quap)
        Rsqs[k] = brain_Rsq(quap)
        print(f"R² = {Rsqs[k]:.4f}")


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
