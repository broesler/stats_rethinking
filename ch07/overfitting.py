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
# FIXME why is this method incorrect? If we can figure this out, we don't need
# to write the sim function to correspond to link! It is just the lmeval with
# a different output variable.
# I believe this method gives an incorrect answer because it does not touch the
# internal rng state of pytensor for each draw, so the draws are *not*
# independent. Thus, the result is a bunch of random points within the PI of
# brain_std, but whose *means* are not necessarily near the brain_std mean.
# h_samp = sts.lmeval(m7_1,
#                     out=m7_1.model.brain_std,
#                     params=[m7_1.model.α,
#                             m7_1.model.β,
#                             m7_1.model.log_σ]
#                     )
h_mean = h_samp.mean(axis=1)
h_pi = sts.percentiles(h_samp, q=0.89, axis=1) * df['brain'].max()
idx = np.argsort(df['mass'])  # sort for plotting purposes only

r = h_mean - df['brain_std']  # residuals
resid_var = r.var(ddof=0)     # pandas default is ddof=1 which uses N-1.
outcome_var = df['brain_std'].var(ddof=0)
Rsq = 1 - resid_var / outcome_var
print(f"{Rsq = :.4f}")

# Plot the data
fig = plt.figure(1, clear=True, constrained_layout=True)
ax = fig.add_subplot()
# TODO label points with species
ax.scatter('mass', 'brain', data=df)
ax.scatter(df['mass'], h_mean * df['brain'].max(), c='k', marker='x')
ax.fill_between(df['mass'].iloc[idx], h_pi[0, idx], h_pi[1, idx],
                facecolor='C0', alpha=0.3, interpolate=True)
ax.set(xlabel='body mass [kg]',
       ylabel='brain volume [cc]')

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
