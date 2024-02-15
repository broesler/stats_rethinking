#!/usr/bin/env python3
# =============================================================================
#     File: ch13H1.py
#  Created: 2024-02-15 11:08
#   Author: Bernie Roesler
#
"""
Exercise 13H1. Bengali contraception.

See also Lecture 13 (2023):
<https://youtu.be/sgqMkZeslxA?si=stxwJaTypUYXcI0R>

The lecture walks through the model more specifically than in the homework,
including some practical coding pitfalls.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pymc as pm
import xarray as xr

from pathlib import Path
from scipy import stats
from scipy.special import expit

import stats_rethinking as sts


df = pd.read_csv(
    Path('../data/bangladesh.csv'),
    dtype=dict({
        'urban': 'bool',
        'use.contraception': 'bool'
    })
).rename({'use.contraception': 'use_contraception'}, axis='columns')

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1934 entries, 0 to 1933
# Data columns (total 6 columns):
#    Column             Non-Null Count  Dtype
# ---  ------             --------------  -----
#  0   woman              1934 non-null   int64
#  1   district           1934 non-null   int64
#  2   use.contraception  1934 non-null   bool
#  3   living.children    1934 non-null   int64
#  4   age.centered       1934 non-null   float64
#  5   urban              1934 non-null   bool
# dtypes: bool(2), float64(1), int64(3)
# memory usage: 64.3 KB

# District 54 is missing:
# df['district'].value_counts().sort_index()

# Ensure District 54 is included in the categories
districts = np.arange(1, df['district'].max()+1)
df['district'] = pd.Categorical(df['district'], categories=districts)

# Make a model of contraception by district
with pm.Model():
    D = pm.ConstantData('D', df['district'].astype(int) - 1)  # 0-indexed
    α_bar = pm.Normal('α_bar', 0, 1)
    σ = pm.Exponential('σ', 1)
    α = pm.Normal('α', α_bar, σ, shape=districts.shape)
    p = pm.Deterministic('p', pm.math.invlogit(α[D]))
    C = pm.Bernoulli('C', p, observed=df['use_contraception'])
    mCD = sts.ulam(data=df)

print('mCD:')
sts.precis(mCD)

# -----------------------------------------------------------------------------
#         Visualize the Data
# -----------------------------------------------------------------------------
counts = df.groupby('district')['woman'].count()
props = df.groupby('district')['use_contraception'].sum() / counts

fig, ax = plt.subplots(num=1, clear=True)
ax.bar(districts, counts.values, color='C3', width=0.5)
ax.set(xlabel='district',
       ylabel='number of women')


def plot_model(model, urban=None, ax=None):
    """Plot the data and posterior predictions."""
    if ax is None:
        ax = plt.gca()

    a = (1 - 0.89) / 2

    if urban is None:
        # Use all of the data
        counts = df.groupby('district')['woman'].count()
        props = df.groupby('district')['use_contraception'].sum() / counts
        p_samp = expit(model.samples['α'])
    else:
        # Filter by urban/rural status
        tf = df.loc[df['urban'] == urban]
        counts = tf.groupby('district')['woman'].count()
        props = tf.groupby('district')['use_contraception'].sum() / counts
        p_samp = expit(
            model.samples['α'] 
            + (model.samples['β'].rename(dict({'β_dim_0': 'α_dim_0'}))
               * urban)
        )

    p_est = p_samp.mean(('chain', 'draw'))
    p_PI = p_samp.quantile([a, 1-a], dim=('chain', 'draw'))

    ax.scatter(districts, props, c='k', alpha=0.8)
    ax.errorbar(districts, p_est, yerr=np.abs(p_PI - p_est),
                ls='none', marker='o', lw=3, alpha=0.6,
                c='C0' if urban else 'C3')

    ax.set(xlabel='district',
           ylabel='prob. use contraception',
           ylim=(-0.05, 1.05))

    return ax


# Plot the data and model results
fig, ax = plt.subplots(num=2, clear=True)
fig.set_size_inches((10, 5), forward=True)
ax.set(title='Proportion by District Only')
plot_model(mCD, ax=ax)

# Label sample sizes of K largest and smallest proportions
K = 5
ext_props = props.loc[~props.isna()].sort_values() 
# Proportions (y-value)
lo_p = ext_props[:K]
hi_p = ext_props[-K:]

# District (x-value)
lo_d = lo_p.index
hi_d = hi_p.index

# Text to display
lo_N = counts.loc[lo_p.index]
hi_N = counts.loc[hi_p.index]

for d, p, N in zip(lo_d, lo_p, lo_N):
    ax.text(x=d + 0.5, y=p, s=str(N), ha='left', va='center')

for d, p, N in zip(hi_d, hi_p, hi_N):
    ax.text(x=d + 0.5, y=p, s=str(N), ha='left', va='center')

ax.annotate(
    'No data!',
    xy=(54, 0.7),
    xytext=(0, 50),
    textcoords='offset points',
    ha='center', 
    va='bottom',
    arrowprops=dict(arrowstyle='->', lw=2)
)


# -----------------------------------------------------------------------------
#         Build a model with urban
# -----------------------------------------------------------------------------
with pm.Model():
    D = pm.ConstantData('D', df['district'].astype(int) - 1)  # 0-indexed
    U = pm.ConstantData('U', df['urban'].astype(int))
    α_bar = pm.Normal('α_bar', 0, 1)
    β_bar = pm.Normal('β_bar', 0, 1)
    σ = pm.Exponential('σ', 1)
    τ = pm.Exponential('τ', 1)
    α = pm.Normal('α', α_bar, σ, shape=districts.shape)
    β = pm.Normal('β', β_bar, σ, shape=districts.shape)
    p = pm.Deterministic('p', pm.math.invlogit(α[D] + β[D]*U))
    C = pm.Bernoulli('C', p, observed=df['use_contraception'])
    mCDU = sts.ulam(data=df)


fig, axs = plt.subplots(num=3, nrows=2, sharex=True, clear=True)
fig.set_size_inches((10, 8), forward=True)

for u, ax in enumerate(axs):
    plot_model(mCDU, urban=u, ax=ax)
    ax.set_title('Urban' if u else 'Rural')

axs[0].set_xlabel('')

# =============================================================================
# =============================================================================
