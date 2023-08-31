#!/usr/bin/env python3
# =============================================================================
#     File: ch07_hard.py
#  Created: 2023-08-30 15:39
#   Author: Bernie Roesler
#
"""
Hard exercises from Ch. 7.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

df = pd.read_csv('../data/Howell1.csv')

# >>> df.info
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 544 entries, 0 to 543
# Data columns (total 4 columns):
#    Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   height  544 non-null    float64
#  1   weight  544 non-null    float64
#  2   age     544 non-null    float64
#  3   male    544 non-null    int64
# dtypes: float64(3), int64(1)
# memory usage: 17.1 KB

df['A'] = sts.standardize(df['age'])

train = df.sample(frac=0.5, random_state=1000)
test = df.drop(train.index)

assert len(train) == 272

# Plot the data
fig, ax = plt.subplots(num=1, clear=True, constrained_layout=True)
ax.scatter('A', 'height', data=df, alpha=0.4)
ax.set(xlabel='age [std]',
       ylabel='height [cm]')


def poly_model(poly_order, x='A', y='height', data=train):
    """Build a polynomial model of the height ~ age relationship."""
    with pm.Model():
        ind = pm.MutableData('ind', data[x])
        X = sts.design_matrix(ind, poly_order)  # [1 x x² x³ ...]
        α = pm.Normal('α', data[y].mean(), 2*data[y].std(), shape=(1,))
        βn = pm.Normal('βn', 0, 100, shape=(poly_order,))
        β = pm.math.concatenate([α, βn])
        μ = pm.Deterministic('μ', pm.math.dot(X, β))
        σ = pm.LogNormal('σ', 0, 1)
        h = pm.Normal('h', μ, σ, observed=data[y], shape=ind.shape)
        # Compute the quadratic posterior approximation
        quap = sts.quap(data=data)
    return quap


# Create polynomial models of height ~ age.
models = {i: poly_model(i, data=train) for i in range(1, 7)}

# -----------------------------------------------------------------------------
#         Prior predictive checks with the linear model
# -----------------------------------------------------------------------------
N = 20
with models[1].model:
    # pm.set_data({'ind': 
    idata = pm.sample_prior_predictive(N)

fig, ax = plt.subplots(num=2, clear=True, constrained_layout=True)
ax.axhline(0, c='k', ls='--', lw=1)   # x-axis
ax.axhline(272, c='k', ls='-', lw=1)  # Wadlow line
ax.plot(train['A'], idata.prior['μ'].mean('chain').T, 'k', alpha=0.4)
ax.set(xlabel='age [std]',
       ylabel='height [cm]')


# 6H1: Compare the models using WAIC
cmp = sts.compare(models.values(), mnames=models.keys())
ct = cmp['ct']
print(ct)
fig, ax = sts.plot_compare(ct, fignum=3)
ax.set_xlabel('Polynomial order')

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
