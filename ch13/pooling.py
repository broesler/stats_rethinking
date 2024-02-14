#!/usr/bin/env python3
# =============================================================================
#     File: pooling.py
#  Created: 2024-02-13 17:17
#   Author: Bernie Roesler
#
"""
§13.2 Pooling.

Simulate tadpole data to compare pooling, no pooling, and partial pooling.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pathlib import Path
from scipy import stats
from scipy.special import logit, expit

import stats_rethinking as sts

# (R code 13.7)
a_bar = 1.5
sigma = 1.5
nponds = 60
Nis = np.r_[5, 10, 25, 35]
Ni = np.repeat(Nis, 15)  # sample size in each pond
assert len(Ni) == nponds

# (R code 13.8)
a_pond = stats.norm(a_bar, sigma).rvs(nponds)

# (R code 13.9)
dsim = pd.DataFrame(dict(pond=range(nponds), Ni=Ni, true_a=a_pond))

# Simulate survivors (R code 13.11)
dsim['Si'] = stats.binom(dsim['Ni'], expit(dsim['true_a'])).rvs(nponds)

# Compute no-pooling estimates (R code 13.12)
dsim['p_nopool'] = dsim['Si'] / dsim['Ni']

# Compute partial pooling estimates from a model (R code 13.13)
with pm.Model():
    N = pm.ConstantData('N', dsim['Ni'])
    pond = pm.ConstantData('pond', dsim['pond'])
    α_bar = pm.Normal('α_bar', 0, 1.5)
    σ = pm.Exponential('σ', 1)
    α_pond = pm.Normal('α_pond', α_bar, σ, shape=pond.shape)
    p = pm.Deterministic('p', pm.math.invlogit(α_pond[pond]))
    S = pm.Binomial('S', N, p, observed=dsim['Si'])
    m13_3 = sts.ulam(data=dsim)

# Get the posterior and estimates (R code 13.15)
post = m13_3.get_samples()
dsim['p_partpool'] = expit(post['α_pond'].mean(('chain', 'draw')))

# True values (R code 13.16)
dsim['p_true'] = expit(dsim['true_a'])

# Compute the errors
dsim['nopool_err'] = np.abs(dsim['p_nopool'] - dsim['p_true'])
dsim['partpool_err'] = np.abs(dsim['p_partpool'] - dsim['p_true'])

# Plot the errors
fig, ax = plt.subplots(num=1, clear=True)

for i, s in zip(range(len(Nis)), ['tiny', 'small', 'medium', 'large']):
    # TODO axis transform 0.9
    ax.text(s=f"{s} ponds ({Nis[i]})", x=15*(i+1) - 7.5, y=0.37,
            ha='center', va='bottom')
    if i > 0:
        ax.axvline(i*15 + 0.5, ls='--', c='gray', lw=1)

ax.scatter('pond', 'nopool_err', c='C0', alpha=0.6, data=dsim)
ax.scatter('pond', 'partpool_err', c='C3', alpha=0.6, data=dsim)

for n in Nis:
    df = dsim.loc[dsim['Ni'] == n]
    ax.plot(df['pond'], df['nopool_err'].mean()*np.ones_like(df), c='C0')
    ax.plot(df['pond'], df['partpool_err'].mean()*np.ones_like(df), c='C3')

ax.set(xlabel='pond',
       ylabel='absolute error')
# =============================================================================
# =============================================================================
