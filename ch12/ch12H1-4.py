#!/usr/bin/env python3
# =============================================================================
#     File: 12H1-4.py
#  Created: 2024-01-24 18:50
#   Author: Bernie Roesler
#
"""
Exercises 12H1 through 12H4 (mislabeled in 2ed as 11H1 etc.)
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path
from scipy import stats

import stats_rethinking as sts

df = pd.read_csv(Path('../data/hurricanes.csv'))

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 92 entries, 0 to 91
# Data columns (total 8 columns):
#    Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   name          92 non-null     object
#  1   year          92 non-null     int64
#  2   deaths        92 non-null     int64
#  3   category      92 non-null     int64
#  4   min_pressure  92 non-null     int64    # [millibar]
#  5   damage_norm   92 non-null     int64    # [$]
#  6   female        92 non-null     int64    # [bool]
#  7   femininity    92 non-null     float64  # [1..11] 1 == M, 11 == F
# dtypes: float64(1), int64(6), object(1)
# memory usage: 5.9 KB

# sns.pairplot(df)  # set deaths, damage_norm to log scale -> linear fit

fem_levels = np.arange(1, 12)

# -----------------------------------------------------------------------------
#         12H1. Simple Poisson Model
# -----------------------------------------------------------------------------
# Build a simple Poisson model of deaths
with pm.Model():
    α = pm.Normal('α', 3, 0.5)
    λ = pm.Deterministic('λ', pm.math.exp(α))
    D = pm.Poisson('D', λ, observed=df['deaths'])
    mD = sts.ulam(data=df)

print('Simple model:')
sts.precis(mD)

# λ = np.exp(mD.coef['α']) = 20.658 ~ df['deaths'].mean() = 20.652

# Build a model with femininity as a predictor
with pm.Model():
    F = pm.MutableData('F', df['femininity'])
    α = pm.Normal('α', 3, 0.5)
    β = pm.Normal('β', 0, 0.1)
    λ = pm.Deterministic('λ', pm.math.exp(α + β*F))
    D = pm.Poisson('D', λ, shape=λ.shape, observed=df['deaths'])
    mF = sts.ulam(data=df)

print('Femininity model:')
sts.precis(mF)

# Q: How strong is the association between femininity of name and deaths?
# A: Not strong.

# Q: Which storms does the model fit (retrodict) well?
# A: Storms with mean # of deaths

# Q: Which storms does the model fit (retrodict) poorly?
# A: Storms with large deviations from the mean # of deaths

fig, ax = plt.subplots(num=1, clear=True)
ax.scatter('femininity', 'deaths', data=df, alpha=0.4)
ax.axhline(df['deaths'].mean(), ls='--', lw=1, c='gray', label='mean deaths')

# Plot posterior mean
sts.lmplot(
    mF,
    mean_var=mF.model.λ,
    eval_at=dict(F=np.linspace(1, 11)),
    ax=ax
)

# Plot posterior predictive
sts.lmplot(
    mF,
    mean_var=mF.model.D,
    eval_at=dict(F=np.linspace(1, 11)),
    ax=ax
)

# -----------------------------------------------------------------------------
#         12H2. Gamma-Poisson Model
# -----------------------------------------------------------------------------
with pm.Model():
    F = pm.MutableData('F', df['femininity'])
    α = pm.Normal('α', 3, 0.5)
    β = pm.Normal('β', 0, 0.1)
    λ = pm.Deterministic('λ', pm.math.exp(α + β*F))
    φ = pm.Exponential('φ', 1)
    D = pm.NegativeBinomial('D', mu=λ, alpha=φ, shape=λ.shape,
                            observed=df['deaths'])
    mGF = sts.ulam(data=df)


# Plot posterior mean
sts.lmplot(
    mGF,
    mean_var=mGF.model.λ,
    eval_at=dict(F=np.linspace(1, 11)),
    line_kws=dict(c='C3'),
    fill_kws=dict(fc='C3'),
    ax=ax
)

# Plot posterior predictive
sts.lmplot(
    mGF,
    mean_var=mGF.model.D,
    eval_at=dict(F=np.linspace(1, 11)),
    line_kws=dict(c='none'),
    fill_kws=dict(fc='C3'),
    ax=ax
)

ax.set(xlabel='femininity',
       ylabel='deaths',
       xticks=fem_levels,
       yscale='log')

ax.legend(
    [plt.Line2D([0], [0], color='C0'),
     plt.Line2D([0], [0], color='C3')],
    ['Poisson Model', 'Gamma-Poisson Model']
)

# Q: Can you explain why the association diminished in strength?
# A: The Gamma-Poisson model has an 89% confidence interval that overlaps
# 0 deaths, as opposed to the Poisson model, which has an 89% confidence
# interval that barely drops below 9 deaths. Since nearly 2/3rds of the
# hurricanes have <= 10 deaths, the Poisson model 

# -----------------------------------------------------------------------------
#         12H3. Include an interaction effect
# -----------------------------------------------------------------------------
# Normalize inputs
df['A'] = sts.standardize(np.log(df['damage_norm']))
df['P'] = sts.standardize(df['min_pressure'])

with pm.Model():
    F = pm.MutableData('F', df['femininity'])
    P = pm.MutableData('P', df['P'])
    A = pm.MutableData('A', df['A'])
    α = pm.Normal('α', 3, 0.5)
    β_F = pm.Normal('β_F', 0, 0.1)
    β_P = pm.Normal('β_P', 0, 0.1)
    β_A = pm.Normal('β_A', 0, 0.1)
    λ = pm.Deterministic('λ', pm.math.exp(α + β_F*F + β_P*P + β_A*A))
    φ = pm.Exponential('φ', 1)
    D = pm.NegativeBinomial('D', mu=λ, alpha=φ, shape=λ.shape,
                            observed=df['deaths'])
    mFPA = sts.ulam(data=df)


# =============================================================================
# =============================================================================
