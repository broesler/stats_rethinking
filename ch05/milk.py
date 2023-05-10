#!/usr/bin/env python3
# =============================================================================
#     File: milk.py
#  Created: 2023-05-08 22:24
#   Author: Bernie Roesler
#
"""
Description: §5.2 Masked Relationships.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from copy import deepcopy
from pathlib import Path
from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

# -----------------------------------------------------------------------------
#        Load Dataset (R code 5.18)
# -----------------------------------------------------------------------------
data_path = Path('../data/')
data_file = Path('milk.csv')

df = pd.read_csv(data_path / data_file)

# >>> df.ino()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 29 entries, 0 to 28
# Data columns (total 8 columns):
#  #   Column          Non-Null Count  Dtype
# ---  ------          --------------  -----
#  0   clade           29 non-null     object
#  1   species         29 non-null     object
#  2   kcal.per.g      29 non-null     float64
#  3   perc.fat        29 non-null     float64
#  4   perc.protein    29 non-null     float64
#  5   perc.lactose    29 non-null     float64
#  6   mass            29 non-null     float64
#  7   neocortex.perc  17 non-null     float64
# dtypes: float64(6), object(2)
# memory usage: 1.9+ KB

# standardize variables (R code 5.19)
df['K'] = sts.standardize(df['kcal.per.g'])
df['N'] = sts.standardize(df['neocortex.perc'])
df['M'] = sts.standardize(np.log(df['mass']))

df = df.dropna(subset=['K', 'N', 'M'])  # (R code 5.22)

# K ~ N (R code 5.20)
with pm.Model() as m5_5_draft:
    ind = pm.MutableData('ind', df['N'])
    α = pm.Normal('α', 0, 1)
    β_N = pm.Normal('β_N', 0, 1)
    σ = pm.Exponential('σ', 1)
    μ = pm.Deterministic('μ', α + β_N * ind)
    K = pm.Normal('K', μ, σ, observed=df['K'], shape=ind.shape)
    # Compute the posterior
    quapNK = sts.quap()

# Sample the priors
Nl = 50
x_s = np.r_[-2, 2]

with m5_5_draft:
    pm.set_data({'ind': x_s})
    prior5_5_draft = pm.sample_prior_predictive(Nl).prior.mean('chain')

# A better prior
with pm.Model() as m5_5:
    ind = pm.MutableData('ind', df['N'])
    obs = pm.MutableData('obs', df['K'])
    α = pm.Normal('α', 0, 0.2)
    β_N = pm.Normal('β_N', 0, 0.5)
    σ = pm.Exponential('σ', 1)
    μ = pm.Deterministic('μ', α + β_N * ind)
    K = pm.Normal('K', μ, σ, observed=obs, shape=ind.shape)
    # Compute the posterior
    quapNK = sts.quap()
    # Compute the prior on a different input
    pm.set_data({'ind': x_s})
    prior5_5 = pm.sample_prior_predictive(Nl).prior.mean('chain')

# Figure 5.7 (R code 5.24)
fig = plt.figure(1, clear=True, constrained_layout=True)
gs = fig.add_gridspec(nrows=1, ncols=2)
ax = fig.add_subplot(gs[0])
ax.plot(np.tile(x_s, (Nl, 1)).T, prior5_5_draft['μ'].T, 'k', alpha=0.4)
ax.set(title=(r'$\alpha \sim \mathcal{N}(0, 1)$'
              + "\n" + r'$\beta_N \sim \mathcal{N}(0, 1)$'),
       xlabel='Neocortex Percent [std]', xlim=x_s,
       ylabel='Mass [kCal/g] [std]', ylim=x_s,
       aspect='equal')

ax = fig.add_subplot(gs[1], sharex=ax, sharey=ax)
ax.plot(np.tile(x_s, (Nl, 1)).T, prior5_5['μ'].T, 'k', alpha=0.4)
ax.set(title=(r'$\alpha \sim \mathcal{N}(0, 0.2)$'
              + "\n" + r'$\beta_N \sim \mathcal{N}(0, 0.5)$'),
       xlabel='Neocortex Percent [std]',
       ylabel=None,
       aspect='equal')
ax.tick_params(axis='y', labelleft=None)

# -----------------------------------------------------------------------------
#         Figure 5.8
# -----------------------------------------------------------------------------
# (R code 5.28, model m5.6)
m5_6 = deepcopy(m5_5)
with m5_6:
    pm.set_data({'ind': df['M'],
                 'obs': df['K']})
    quapMK = sts.quap()
    quapMK.rename({'β_N': 'β_M'})
    m5_6.β_N.name = 'β_M'

print('K ~ N:')
sts.precis(quapNK)
print('K ~ M:')
sts.precis(quapMK)

# Plot the regressions
fig = plt.figure(2, clear=True, tight_layout=True)
fig.set_size_inches((6, 6), forward=True)
gs = fig.add_gridspec(nrows=2, ncols=2)
ax = fig.add_subplot(gs[0, 0])
sts.lmplot(quapNK, mean_var=quapNK.model.μ, data=df, x='N', y='K', ax=ax)
ax.set(xlabel='Neocortex Percent [std]',
       ylabel='Mass [kCal/g] [std]',
       xlim=x_s, ylim=x_s)

ax = fig.add_subplot(gs[0, 1], sharey=ax)
sts.lmplot(quapMK, mean_var=quapMK.model.μ, data=df, x='M', y='K', ax=ax)
ax.set(xlabel='Log(Body Mass) [std]',
       ylabel=None)
ax.tick_params(axis='y', labelleft=None)

# Compute the full model (R code 5.29)
with pm.Model() as m5_7:
    N = pm.MutableData('N', df['N'])
    M = pm.MutableData('M', df['M'])
    α = pm.Normal('α', 0, 0.2)
    β_N = pm.Normal('β_N', 0, 0.5)
    β_M = pm.Normal('β_M', 0, 0.5)
    σ = pm.Exponential('σ', 1)
    μ = pm.Deterministic('μ', α + β_N * N + β_M * M)
    K = pm.Normal('K', μ, σ, observed=df['K'], shape=N.shape)
    # Compute the posterior
    quap = sts.quap()

print('K ~ M + N:')
sts.precis(quap)

# Plot counterfactuals
q = 0.89
N_s = np.linspace(-2, 2, 30)

# Plot counterfactual with M = 0 (R code 5.31)
ax = fig.add_subplot(gs[1, 0], sharex=fig.axes[0], sharey=ax)
sts.lmplot(quap, mean_var=μ, x='N', y='K',
           eval_at={'N': N_s, 'M': np.zeros_like(N_s)}, ax=ax)
ax.set(title='Counterfactual, M = 0',
       xlabel='Neocortex Percent [std]',
       ylabel='Mass [kCal/g] [std]')

# Plot counterfactual with N = 0
ax = fig.add_subplot(gs[1, 1], sharex=fig.axes[1], sharey=ax)
sts.lmplot(quap, mean_var=μ, x='M', y='K',
           eval_at={'N': np.zeros_like(N_s), 'M': N_s}, ax=ax)
ax.set(title='Counterfactual, N = 0',
       xlabel='Log(Body Mass) [std]',
       ylabel=None)
ax.tick_params(axis='y', labelleft=None)

for ax in fig.axes:
    ax.set_aspect('equal')

# (R code 5.30)
ct = sts.coef_table(models=[quapMK, quapNK, quap],
                    mnames=['m5.5', 'm5.6', 'm5.7'],
                    params=['β_M', 'β_N']
                    )

fig = plt.figure(3, clear=True, constrained_layout=True)
fig.set_size_inches((5, 3), forward=True)
ax = fig.add_subplot()
sts.plot_coef_table(ct, ax=ax)

sns.pairplot(data=df, vars=['M', 'N', 'K'])

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
