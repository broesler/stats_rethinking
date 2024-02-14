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
post = m13_2.get_samples()
p_est = m13_2.deterministics['p'].mean(('chain', 'draw'))

a = (1 - 0.89) / 2
p_PI = m13_2.deterministics['p'].quantile([a, 1-a], dim=('chain', 'draw'))

fig, ax = plt.subplots(num=1, clear=True)

# Label tank sizes
ax.axvline(15.5, ls='--', c='gray', lw=1)
ax.axvline(31.5, ls='--', c='gray', lw=1)
ax.text(     8, 0, 'small tanks (10)',  ha='center', va='bottom')
ax.text(15 + 8, 0, 'medium tanks (25)', ha='center', va='bottom')
ax.text(31 + 8, 0, 'large tanks (35)',  ha='center', va='bottom')

# Plot the data mean and prediction parameter mean
ax.axhline(df['propsurv'].mean(), ls='--', c='k', lw=1)
ax.axhline(expit(post['α_bar'].mean()), ls='--', c='C3', lw=1)

# Plot the data with a percentile interval
ax.errorbar(
    df['tank'],
    p_est,
    yerr=np.abs(p_PI - p_est),
    marker='o', ls='none', c='C3', lw=3, alpha=0.7,
)

# Plot the data and predictions
ax.scatter('tank', 'propsurv', data=df, c='k', label='data', zorder=3)

ax.set(xlabel='tank',
       ylabel='proportion survival',
       ylim=(-0.05, 1.05))

# -----------------------------------------------------------------------------
#         Plot first 100 populations in posterior (R code 13.6)
# -----------------------------------------------------------------------------
N_lines = 100
xs = np.linspace(-3, 4)

sim_tanks = stats.norm(post['α_bar'], post['σ']).rvs().flatten()

fig, axs = plt.subplots(num=2, ncols=2, clear=True)
fig.set_size_inches((10, 5), forward=True)

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


# -----------------------------------------------------------------------------
#         Add predator predictor
# -----------------------------------------------------------------------------
# See Lecture 12 (2023) @ 43:52
with pm.Model():
    N = pm.ConstantData('N', df['density'])
    P = pm.ConstantData('P', df['pred'].cat.codes)
    tank = pm.ConstantData('tank', df['tank'])
    α_bar = pm.Normal('α_bar', 0, 1.5)
    β_P = pm.Normal('β_P', 0, 0.5)
    σ = pm.Exponential('σ', 1)
    α = pm.Normal('α', α_bar, σ, shape=tank.shape)
    p = pm.Deterministic('p', pm.math.invlogit(α[tank] + β_P * P))
    S = pm.Binomial('S', N, p, observed=df['surv'])
    mSTP = sts.ulam(data=df)

mST = m13_2  # match naming in lecture

post_ST = mST.get_samples()
post_STP = mSTP.get_samples()

surv_ST = (
    pm.sample_posterior_predictive(post_ST, model=mST.model)
    .posterior_predictive['S']
    .mean(('chain', 'draw'))
) / df['density']

surv_STP = (
    pm.sample_posterior_predictive(post_STP, model=mSTP.model)
    .posterior_predictive['S']
    .mean(('chain', 'draw'))
) / df['density']


fig, axs = plt.subplots(num=3, ncols=2, clear=True)
# Compare the two model predictions
ax = axs[0]
ax.axline((0, 0), (1, 1), ls='--', c='gray', lw=1)
ax.scatter(surv_ST, surv_STP, c=np.where(df['pred'] == 'pred', 'C3', 'C0'))
ax.set(xlabel='prob survival (model without predators)',
       ylabel='prob survival (model with predators)',
       aspect='equal')
ax.spines[['top', 'right']].set_visible(False)

# Compare the hyperparameter distributions
ax = axs[1]
sns.kdeplot(post_ST['σ'].values.flat, 
            bw_adjust=0.5, color='C0', ax=ax, label='mST')
sns.kdeplot(post_STP['σ'].values.flat, 
            bw_adjust=0.5, color='C3', ax=ax, label='mSTP')
ax.legend()
ax.set_xlabel('σ')
ax.spines[['top', 'right']].set_visible(False)


# -----------------------------------------------------------------------------
#         Plot model results
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(num=4, clear=True)

# Label tank sizes
ax.axvline(15.5, ls='--', c='gray', lw=1)
ax.axvline(31.5, ls='--', c='gray', lw=1)
ax.text(     8, 0, 'small tanks (10)',  ha='center', va='bottom')
ax.text(15 + 8, 0, 'medium tanks (25)', ha='center', va='bottom')
ax.text(31 + 8, 0, 'large tanks (35)',  ha='center', va='bottom')

ax.set(xlabel='tank',
       ylabel='proportion survival',
       ylim=(-0.05, 1.05))
ax.spines[['top', 'right']].set_visible(False)

# Plot the data
ax.scatter('tank', 'propsurv', data=df, c='k', label='data')

for p, c in zip(['no', 'pred'], ['C0', 'C3']):
    tf = df.loc[df['pred'] == p]
    ax.errorbar(
        tf['tank'],
        p_est.sel(p_dim_0=tf.index),
        yerr=np.abs(p_PI - p_est).sel(p_dim_0=tf.index),
        marker='o', ls='none', c=c, lw=3, alpha=0.7,
        label='no predators' if p == 'no' else 'predators present',
    )

ax.legend()

# =============================================================================
# =============================================================================
