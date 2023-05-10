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

# 5H1. Two bivariate regressions

# (1) body weight ~ territory size (area)
with pm.Model() as mWA:
    ind = pm.MutableData('ind', df['A'])
    α = pm.Normal('α', 0, 0.2)
    β = pm.Normal('β', 0, 0.5)
    μ = pm.Deterministic('μ', α + β * ind)
    σ = pm.Exponential('σ', 1)
    W = pm.Normal('W', μ, σ, observed=df['W'])
    quapWA = sts.quap(data=df)

print('W ~ A:')
sts.precis(quapWA)

# Plot the regressions
fig = plt.figure(1, clear=True, tight_layout=True)
ax = fig.add_subplot()
sts.lmplot(quapWA, mean_var=quapWA.model.μ, 
           data=df, x='area', y='weight', unstd=True, ax=ax)
ax.set(xlabel='Territory Area [std]',
       ylabel='Body Weight [std]')

with mWA:
    pm.set_data({'ind': df['G']})
    quapWG = sts.quap(data=df)

print('W ~ G:')
sts.precis(quapWG)

# Plot the regressions
fig = plt.figure(2, clear=True, tight_layout=True)
ax = fig.add_subplot()
sts.lmplot(quapWG, mean_var=quapWG.model.μ, 
           data=df, x='groupsize', y='weight', unstd=True, ax=ax)
ax.set(xlabel='Group Size [std]',
       ylabel='Body Weight [std]')

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
