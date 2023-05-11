#!/usr/bin/env python3
# =============================================================================
#     File: ch05_hard.py
#  Created: 2023-05-10 16:48
#   Author: Bernie Roesler
#
"""
Description: Solutions to Hard exercises.
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
data_file = Path('foxes.csv')

df = pd.read_csv(data_path / data_file)

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 116 entries, 0 to 115
# Data columns (total 5 columns):
#    Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   group      116 non-null    int64
#  1   avgfood    116 non-null    float64
#  2   groupsize  116 non-null    int64
#  3   area       116 non-null    float64
#  4   weight     116 non-null    float64
# dtypes: float64(3), int64(2)
# memory usage: 4.7 KB

# Standardize variables
df['W'] = sts.standardize(df['weight'])
df['A'] = sts.standardize(df['area'])
df['G'] = sts.standardize(df['groupsize'])  # assume groupsize is continuous?

# Exploratory plot
g = sns.PairGrid(df, vars=['A', 'G', 'W'],  corner=True)
g.map_diag(sns.histplot)
g.map_lower(sns.regplot)
# A ~ G strongly correlated

# ----------------------------------------------------------------------------- 
#         5H1. Two bivariate regressions
# -----------------------------------------------------------------------------

# (1) body weight ~ territory size (area)
with pm.Model() as mWA:
    x = pm.MutableData('x', df['A'])
    α = pm.Normal('α', 0, 0.2)
    β = pm.Normal('β', 0, 0.5)
    μ = pm.Deterministic('μ', α + β * x)
    σ = pm.Exponential('σ', 1)
    W = pm.Normal('W', μ, σ, observed=df['W'], shape=x.shape)
    quapWA = sts.quap(data=df)

print('W ~ A:')
sts.precis(quapWA)

# Plot the regressions
fig = plt.figure(1, clear=True, tight_layout=True)
gs = fig.add_gridspec(nrows=1, ncols=2)
ax = fig.add_subplot(gs[0])
sts.lmplot(quapWA, mean_var=quapWA.model.μ, 
           data=df, x='area', y='weight', unstd=True, q=0.95, ax=ax)
ax.set(xlabel='Territory Area',
       ylabel='Body Weight')

# (2) body weight ~ group size
with mWA:
    pm.set_data({'x': df['G']})
    quapWG = sts.quap(data=df)

print('W ~ G:')
sts.precis(quapWG)

# β_G is nearly negative, but close to 0.

# Plot the regressions
ax = fig.add_subplot(gs[1])
sts.lmplot(quapWG, mean_var=quapWG.model.μ, 
           data=df, x='groupsize', y='weight', unstd=True, q=0.95, ax=ax)
ax.set(xlabel='Group Size',
       ylabel=None)
ax.tick_params(axis='y', labelleft=None)

# ----------------------------------------------------------------------------- 
#         5H2. Multiple linear regression
# -----------------------------------------------------------------------------
with pm.Model() as mWA:
    A = pm.MutableData('A', df['A'])
    G = pm.MutableData('G', df['G'])
    α = pm.Normal('α', 0, 0.2)
    β_A = pm.Normal('β_A', 0, 0.5)
    β_G = pm.Normal('β_G', 0, 0.5)
    μ = pm.Deterministic('μ', α + β_A * A + β_G * G)
    σ = pm.Exponential('σ', 1)
    W = pm.Normal('W', μ, σ, observed=df['W'], shape=A.shape)
    quap = sts.quap(data=df)

print('W ~ A + G:')
sts.precis(quap)

A_s = np.linspace(-2.5, 2.5, 30)

# Plot counterfactual with M = 0 (R code 5.31)
fig = plt.figure(2, clear=True, constrained_layout=True)
gs = fig.add_gridspec(nrows=1, ncols=2)
ax = fig.add_subplot(gs[0])
sts.lmplot(quap, mean_var=μ, x='A', y='W',
           eval_at={'A': A_s, 'G': np.zeros_like(A_s)}, ax=ax)
ax.set(title='Counterfactual, G = 0',
       xlabel='Territory Area [std]',
       ylabel='Body Weight [std]')

# Plot counterfactual with N = 0
ax = fig.add_subplot(gs[1], sharey=ax)
sts.lmplot(quap, mean_var=μ, x='G', y='W',
           eval_at={'A': np.zeros_like(A_s), 'G': A_s}, ax=ax)
ax.set(title='Counterfactual, A = 0',
       xlabel='Group Size [std]',
       ylabel=None)
ax.tick_params(axis='y', labelleft=None)

for ax in fig.axes:
    ax.set_aspect('equal')

# (R code 5.30)
quapWA.rename({'β': 'β_A'})
quapWG.rename({'β': 'β_G'})
ct = sts.coef_table(models=[quapWA, quapWG, quap],
                    mnames=['W ~ A', 'W ~ G', 'W ~ G + A'],
                    params=['β_A', 'β_G']
                    )

sts.plot_coef_table(ct, fignum=3)

# ----------------------------------------------------------------------------- 
#         5H3. One more variable.
# -----------------------------------------------------------------------------

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
