#!/usr/bin/env python3
# =============================================================================
#     File: 11H4.py
#  Created: 2024-01-11 20:52
#   Author: Bernie Roesler
#
"""
Solution to 11H4. Salamanders data.

.. note:: Problem is mislabeled as 10H4 in the digital version.

The model will be of the number of salamanders as a Poisson model.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path
from scipy import stats

import stats_rethinking as sts

df = pd.read_csv(Path('../data/salamanders.csv'))

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 47 entries, 0 to 46
# Data columns (total 4 columns):
#    Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   SITE       47 non-null     int64
#  1   SALAMAN    47 non-null     int64  # counts of salamanders
#  2   PCTCOVER   47 non-null     int64  # percent ground cover
#  3   FORESTAGE  47 non-null     int64  # age of trees in 49-m² plot
# dtypes: int64(4)
# memory usage: 1.6 KB

df = df.set_index('SITE').drop(30)  # FORESTAGE == 0
df['S'] = df['SALAMAN']
df['P'] = sts.standardize(df['PCTCOVER'])
df['A'] = sts.standardize(np.log(df['FORESTAGE']))

S_mean = df['S'].mean()  # ~ 2.5

# Plot the prior for λ with a given intercept
xe = np.linspace(0, 10, 200)
σ = 0.5
μ = np.log(S_mean) + σ**2  # want the *mode* of the lognormal to be S_mean
α_strong = stats.lognorm.pdf(xe, scale=np.exp(μ), s=σ)

fig, ax = plt.subplots(num=1, clear=True)
ax.axvline(S_mean, c='C0', ls='-.', lw=1, alpha=0.5, label='desired mean')
ax.axvline(np.exp(μ - σ**2), c='k', ls='--', lw=1, alpha=0.5, label='mode')
ax.axvline(np.exp(μ + σ**2 / 2), c='k', ls=':', lw=1, alpha=0.5, label='mean')
ax.plot(xe, α_strong)
ax.set_title((
    r'$\lambda = e^\alpha \text{ with } \alpha \sim '
    rf"\mathcal{{N}}({μ:.2f}, {σ:.2f})$"
))
ax.set_xlabel('Mean Number of Salamanders')

# Plot prior predictive
# Simulate a prior with a slope
N = 100
xs = np.linspace(-2, 2, 200)

α_prior = stats.norm(μ, σ).rvs(N)
β_prior = stats.norm(0, 0.25).rvs(N)

def λ_prior(x):
    return np.exp(α_prior + β_prior*np.c_[x])

fig, axs = plt.subplots(num=2, ncols=2, sharey=True, clear=True)
fig.set_size_inches((10, 5), forward=True)

# plot x-axis on log-standard scale 
axs[0].axhline(df['S'].max(), c='k', ls='--', lw=1, alpha=0.3)
axs[0].plot(xs, λ_prior(xs), c='k', alpha=0.3)
axs[0].set(xlabel='% cover [std]',
           ylabel='Number of Salamanders',
           ylim=(0, 30),
           xticks=np.arange(-2, 3))
axs[0].spines[['top', 'right']].set_visible(False)

# plot x-axis on natural scale 
x = np.linspace(0, 100, 200)
axs[1].axhline(df['S'].max(), c='k', ls='--', lw=1, alpha=0.3)
axs[1].plot(x, λ_prior(x), c='k', alpha=0.3)
axs[1].set(xlabel='% cover')
axs[1].spines[['top', 'right']].set_visible(False)

# ----------------------------------------------------------------------------- 
#         (a) Model the salamander counts as a Poisson model
# -----------------------------------------------------------------------------
with pm.Model():
    α = pm.Normal('α', μ, σ)
    β = pm.Normal('β', 0, 0.5)
    λ = pm.Deterministic('λ', pm.math.exp(α + β * df['P']))
    y = pm.Poisson('y', λ, observed=df['S'])
    m_quap = sts.quap()
    m_ulam = sts.ulam(data=df)

print('quap:')
sts.precis(m_quap)
print('ulam:')
sts.precis(m_ulam)

with m_ulam.model:
    post = m_ulam.get_samples()
    post_y = pm.sample_posterior_predictive(post).posterior_predictive['y']

post_λ = m_ulam.deterministics['λ']

x = df['PCTCOVER'].copy()
idx = np.argsort(x)
x = x.iloc[idx]

# Plot predictions vs PCTCOVER
fig, ax = plt.subplots(num=1, clear=True)
sts.lmplot(
    fit_x=x,
    fit_y=post_λ.isel(λ_dim_0=idx),
    x='PCTCOVER',
    y='SALAMAN',
    data=df,
    ax=ax,
)
sts.lmplot(
    fit_x=x,
    fit_y=post_y.isel(y_dim_2=idx),
    ax=ax,
)

ax.set(xlabel='% cover',
       ylabel='Number of Salamanders')
ax.spines[['top', 'right']].set_visible(False)


# ----------------------------------------------------------------------------- 
#         (b) Improve the model with FORESTAGE
# -----------------------------------------------------------------------------
with pm.Model():
    α = pm.Normal('α', μ, σ)
    β_P = pm.Normal('β_P', 0, 0.5)
    β_A = pm.Normal('β_A', 0, 0.5)
    β_AP = pm.Normal('β_AP', 0, 0.5)
    λ = pm.Deterministic(
        'λ',
        pm.math.exp(
            α 
            + β_P * df['P'] 
            + β_A * df['A'] 
            + β_AP * df['A'] * df['P']
        )
    )
    y = pm.Poisson('y', λ, observed=df['S'])
    m_forest = sts.ulam(data=df)

print('forest:')
sts.precis(m_forest)

cmp = sts.compare([m_ulam, m_forest], ['cover', 'cover + forest'])['ct']
print(cmp)

# FORESTAGE has no effect on prediction! Coefficients are nearly 0.
# Hypothesis: Salamanders' diet is unaffected by old-growth forest.


# =============================================================================
# =============================================================================
