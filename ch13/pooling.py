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

from scipy import stats
from scipy.special import expit

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
df = pd.DataFrame(dict(pond=range(nponds), Ni=Ni, true_a=a_pond))

# Simulate survivors (R code 13.11)
df['Si'] = stats.binom(df['Ni'], expit(df['true_a'])).rvs(nponds)

# Compute no-pooling estimates (R code 13.12)
df['p_nopool'] = df['Si'] / df['Ni']

# Compute partial pooling estimates from a model (R code 13.13)
with pm.Model():
    N = pm.ConstantData('N', df['Ni'])
    pond = pm.ConstantData('pond', df['pond'])
    α_bar = pm.Normal('α_bar', 0, 1.5)
    σ = pm.Exponential('σ', 1)
    α_pond = pm.Normal('α_pond', α_bar, σ, shape=pond.shape)
    p = pm.Deterministic('p', pm.math.invlogit(α_pond[pond]))
    S = pm.Binomial('S', N, p, observed=df['Si'])
    m13_3 = sts.ulam(data=df)

# Get the posterior and estimates (R code 13.15)
post = m13_3.get_samples()
df['p_partpool'] = expit(post['α_pond'].mean(('chain', 'draw')))

# True values (R code 13.16)
df['p_true'] = expit(df['true_a'])

# Compute the errors (R code 13.17)
df['nopool_err'] = np.abs(df['p_nopool'] - df['p_true'])
df['partpool_err'] = np.abs(df['p_partpool'] - df['p_true'])

# Plot the errors (R code 13.18)
fig, ax = plt.subplots(num=1, clear=True)

for i, s in zip(range(len(Nis)), ['tiny', 'small', 'medium', 'large']):
    ax.text(
        s=f"{s} ponds ({Nis[i]})",
        x=15*(i+1) - 7.5,  # data coordinates
        y=1.0,             # axis coordinates (1.0 == top of y-axis limit)
        ha='center',
        va='top',
        transform=ax.get_xaxis_transform()  # x = data, y = axis
    )
    if i > 0:
        ax.axvline(i*15 - 0.5, ls='--', c='gray', lw=1)

ax.scatter('pond', 'nopool_err', c='C0', alpha=0.6, data=df)
ax.scatter('pond', 'partpool_err', c='C3', alpha=0.6, data=df)

for n in Nis:
    tf = df.loc[df['Ni'] == n]
    ax.plot(tf['pond'], np.full_like(tf, tf['nopool_err'].mean()), c='C0')
    ax.plot(tf['pond'], np.full_like(tf, tf['partpool_err'].mean()), c='C3')

ax.legend(loc='center right')
ax.set(xlabel='pond',
       ylabel='absolute error')
ax.spines[['top', 'right']].set_visible(False)

# =============================================================================
# =============================================================================
