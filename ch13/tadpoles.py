#!/usr/bin/env python3
# =============================================================================
#     File: tadpoles.py
#  Created: 2024-02-13 15:11
#   Author: Bernie Roesler
#
"""
§13.1 Multilevel Tadpoles.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from pathlib import Path
from scipy import stats
from scipy.special import expit

import stats_rethinking as sts


# (R code 13.1)
df = pd.read_csv(
    Path('../data/reedfrogs.csv'),
    dtype=dict(pred='category', size='category')
)

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 48 entries, 0 to 47
# Data columns (total 5 columns):
#    Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   density   48 non-null     int64
#  1   pred      48 non-null     category
#  2   size      48 non-null     category
#  3   surv      48 non-null     int64
#  4   propsurv  48 non-null     float64
# dtypes: category(2), float64(1), int64(2)
# memory usage: 1.4 KB

# Approximate the tadpole mortality in each tank (R code 13.2)
df['tank'] = df.index

with pm.Model():
    N = pm.ConstantData('N', df['density'])
    tank = pm.ConstantData('tank', df['tank'])
    α = pm.Normal('α', 0, 1.5, shape=tank.shape)
    p = pm.Deterministic('p', pm.math.invlogit(α[tank]))
    S = pm.Binomial('S', N, p, observed=df['surv'])
    m13_1 = sts.ulam(data=df)

# print('m13.1:')
# sts.precis(m13_1)

# Create multilevel model (R code 13.3)
with pm.Model():
    N = pm.ConstantData('N', df['density'])
    tank = pm.ConstantData('tank', df['tank'])
    α_bar = pm.Normal('α_bar', 0, 1.5)
    σ = pm.Exponential('σ', 1)
    α = pm.Normal('α', α_bar, σ, shape=tank.shape)
    p = pm.Deterministic('p', pm.math.invlogit(α[tank]))
    S = pm.Binomial('S', N, p, observed=df['surv'])
    m13_2 = sts.ulam(data=df)

print('m13.2:')
sts.precis(m13_2, filter_kws=dict(regex='α_bar|σ'))

# (R code 13.4
ct = sts.compare([m13_1, m13_2], ['m13.1', 'm13.2'], sort=True)
print(ct['ct'])

# -----------------------------------------------------------------------------
#         Plot the data (R code 13.5)
# -----------------------------------------------------------------------------
# TODO copy plots from Lecture 12 (2023) showing variance of α
post = m13_2.get_samples()
df['p_est'] = m13_2.deterministics['p'].mean(('chain', 'draw'))

fig, ax = plt.subplots(num=1, clear=True)

# Label tank sizes
ax.axvline(16.5, ls='--', c='gray', lw=1)
ax.axvline(32.5, ls='--', c='gray', lw=1)
ax.axhline(expit(post['α_bar'].mean(('chain', 'draw'))), ls='--', c='k', lw=1)

ax.text(8, 0, s='small tanks', ha='center', va='bottom')
ax.text(16 + 8, 0, 'medium tanks', ha='center', va='bottom')
ax.text(32 + 8, 0, 'large tanks', ha='center', va='bottom')

# Plot the data and predictions
ax.scatter('tank', 'propsurv', data=df, c='k', label='data')
ax.scatter('tank', 'p_est', data=df, c='C3', label='p_est')

ax.set(xlabel='tank',
       ylabel='proportion survival',
       ylim=(-0.05, 1.05))


# Plot first 100 populations in posterior (R code 13.6)
N_lines = 100
xs = np.linspace(-3, 4)

sim_tanks = stats.norm(post['α_bar'], post['σ']).rvs().flatten()

fig, axs = plt.subplots(num=2, ncols=2, clear=True)

μ = post['α_bar'].isel(chain=0, draw=range(N_lines))
s = post['σ'].isel(chain=0, draw=range(N_lines))
pdf = stats.norm(μ, s).pdf(np.c_[xs])
axs[0].plot(xs, pdf, c='k', lw=1, alpha=0.2)
axs[0].set(xlabel='log-odds survival',
           ylabel='Density')

sns.kdeplot(expit(sim_tanks), bw_adjust=0.1, color='k', lw=1, ax=axs[1])
axs[1].set(xlabel='probability survive',
           ylabel='Density')

for ax in axs:
    ax.spines[['top', 'right']].set_visible(False)


# =============================================================================
# =============================================================================
