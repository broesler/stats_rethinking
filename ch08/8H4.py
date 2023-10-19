#!/usr/bin/env python3
# =============================================================================
#     File: 8H4.py
#  Created: 2023-10-03 14:05
#   Author: Bernie Roesler
#
"""
Solution to 8H4.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pathlib import Path

import stats_rethinking as sts

df = pd.read_csv(Path('../data/nettle.csv'))

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 74 entries, 0 to 73
# Data columns (total 7 columns):
#      Column               Non-Null Count  Dtype
# ---  ------               --------------  -----
#  0   country              74 non-null     object
#  1   num.lang             74 non-null     int64
#  2   area                 74 non-null     int64
#  3   k.pop                74 non-null     int64
#  4   num.stations         74 non-null     int64
#  5   mean.growing.season  74 non-null     float64
#  6   sd.growing.season    74 non-null     float64
# dtypes: float64(2), int64(4), object(1)
# memory usage: 4.2 KB

df.columns = df.columns.str.replace('.', '_')

df['lang_per_cap'] = df['num_lang'] / df['k_pop']  # num. languages per capita
df['log_lang'] = np.log(df['lang_per_cap'])

# -----------------------------------------------------------------------------
#         8H4(a)
# -----------------------------------------------------------------------------
# Hypothesis: lang_per_cap ~ mean_growing_season
# Consider log(area) as a covariate, not as an interaction

df['log_area'] = np.log(df['area'])

# TODO re-copy `precis` outputs below to match update to `mean` normalization.
# Standardize variables of interest
df['L'] = df['log_lang'] / df['log_lang'].mean()  # proportion of mean
df['A'] = df['log_area'] / df['log_area'].mean()
df['M'] = df['mean_growing_season'] / df['mean_growing_season'].mean()
df['S'] = df['sd_growing_season'] / df['sd_growing_season'].mean()

# Plot the raw data
fig, ax = plt.subplots(num=1, clear=True)
ax.scatter('M', 'L', s=df['area']*1e-4, data=df, alpha=0.4)
ax.set(xlabel='Mean Growing Season [std]',
       ylabel='log Languages per Capita (prop. of mean)')

# Priors justification
print(f"{df['L'].mean() = :.2f}")  # 1.0
print(f"{df['L'].std() = :.2f}")   # 0.28
print(f"max β_M = {(df['L'].max() - df['L'].min()):.2f}")  # 1.63
max_β_A = ((df['L'].max() - df['L'].min()) / (df['A'].max() - df['A'].min()))
print(f"{max_β_A = :.2f}")  # 3.22
max_β_S = ((df['L'].max() - df['L'].min()) / (df['S'].max() - df['S'].min()))
print(f"{max_β_S = :.2f}")  # 3.22
# Choose α std ≈ std(L)
# Choose β_M std ≈ slope / 3 ≈ 0.5 so max slope is 3 std away
# Choose β_A std ≈ slope / 3 ≈ 1.0
# Choose β_S std ≈ slope / 3 ≈ 1.0

# Build a model L ~ M to observe priors
with pm.Model():
    M, L = (pm.MutableData(x, df[x]) for x in list('ML'))
    α = pm.Normal('α', 1.0, 0.25)
    β_M = pm.Normal('β_M', 0, 0.5)
    μ = pm.Deterministic('μ', α + β_M*M)
    σ = pm.Exponential('σ', 1)
    y = pm.Normal('y', μ, σ, observed=L, shape=M.shape)
    quapM = sts.quap(data=df)

# Plot prior predictive
N_lines = 50
Ms = np.r_[-0.1, 1.1]
with quapM.model:
    pm.set_data({'M': Ms})
    idata = pm.sample_prior_predictive(N_lines)

# Plot priors
fig, ax = plt.subplots(num=2, clear=True)

ax.axhline(df['L'].min(), c='k', ls='--', lw=1)
ax.axhline(df['L'].max(), c='k', ls='--', lw=1)

ax.plot(Ms, idata.prior['μ'].mean('chain').T, c='k', alpha=0.3)

ax.set(
    title=(r'$\alpha \sim \mathcal{N}(1, 0.25)$'
           '\n'
           r'$\beta_M \sim \mathcal{N}(0, 0.5)$'),
    xlabel='Mean Growing Season [std]',
    ylabel='log Languages per Capita (prop. of mean)',
    xlim=(0, 1),
    ylim=(0.0, 2.0),
)

print('L ~ M')
sts.precis(quapM)
#          mean       std      5.5%     94.5%
# α    1.196650  0.069688  1.085274  1.308025
# β_M -0.339954  0.108863 -0.513939 -0.165969
# σ    0.258092  0.021196  0.224216  0.291968

# Build a model L ~ A + M to observe priors
with pm.Model():
    A, M, L = (pm.MutableData(x, df[x]) for x in list('AML'))
    α = pm.Normal('α', 1.0, 0.25)
    β_M = pm.Normal('β_M', 0, 0.5)
    β_A = pm.Normal('β_A', 0, 1)
    μ = pm.Deterministic('μ', α + β_M*M + β_A*A)
    σ = pm.Exponential('σ', 1)
    y = pm.Normal('y', μ, σ, observed=L, shape=M.shape)
    quapMA = sts.quap(data=df)

print('L ~ M + A')
sts.precis(quapMA)
#        mean    std    5.5%   94.5%
# α    0.9055 0.2018  0.5830  1.2279
# β_A  0.2930 0.1905 -0.0114  0.5975
# β_M -0.3360 0.1075 -0.5078 -0.1643  *** negative association!!
# σ    0.2547 0.0209  0.2213  0.2882

# NOTE there is a negative association between mean growing season length and
# log languages per capita, which holds when controlling for area.

# Plot counterfactual posterior predictive results
Ms = np.linspace(-0.1, 1.1)
fig = plt.figure(3, clear=True)
axs = fig.subplots(ncols=2, sharey=True)

# Plot counterfactual across mean growing season
sts.lmplot(
    quap=quapMA, mean_var=quapMA.model.μ,
    x='M', y='L', data=df,
    eval_at={'M': Ms, 'A': np.ones_like(Ms)},
    ax=axs[0],
)

axs[0].set(title='Counterfactual at A = 1.0 (mean)',
       xlim=(0, 1))

# Plot counterfactual across area
As = np.linspace(df['A'].min() - 0.05, df['A'].max() + 0.05)

sts.lmplot(
    quap=quapMA, mean_var=quapMA.model.μ,
    x='A', y='L', data=df,
    eval_at={'M': np.ones_like(As), 'A': As},
    ax=axs[1],
)

axs[1].tick_params(left=False)
axs[1].set(title='Counterfactual at M = 1.0 (mean)', ylabel=None)

# Compare the two models
sts.plot_coef_table(sts.coef_table([quapM, quapMA], ['M', 'M + A']), fignum=4)


# ----------------------------------------------------------------------------- 
#         8H4(b) Effects of uncertainty
# -----------------------------------------------------------------------------
# Build a model L ~ A + S to observe priors
with pm.Model():
    A, S, L = (pm.MutableData(x, df[x]) for x in list('ASL'))
    α = pm.Normal('α', 1.0, 0.25)
    β_S = pm.Normal('β_S', 0, 0.5)
    β_A = pm.Normal('β_A', 0, 1)
    μ = pm.Deterministic('μ', α + β_S*S + β_A*A)
    σ = pm.Exponential('σ', 1)
    y = pm.Normal('y', μ, σ, observed=L, shape=S.shape)
    quapSA = sts.quap(data=df)

print('L ~ S + A')
sts.precis(quapSA)
#       mean    std    5.5%  94.5%
# α   0.7789 0.1989  0.4610 1.0969
# β_A 0.1421 0.2167 -0.2042 0.4884
# β_S 0.2845 0.1777  0.0005 0.5684  *** positive effect!!
# σ   0.2658 0.0219  0.2307 0.3008

# NOTE there is a positive association between std growing season length and
# log languages per capita.

# Plot counterfactual posterior predictive results
fig = plt.figure(5, clear=True)
axs = fig.subplots(ncols=2, sharey=True)

# Plot counterfactual across mean growing season
sts.lmplot(
    quap=quapSA, mean_var=quapSA.model.μ,
    x='S', y='L', data=df,
    eval_at={'S': Ms, 'A': np.ones_like(Ms)},
    ax=axs[0],
)

axs[0].set(title='Counterfactual at A = 1.0 (mean)',
           xlim=(0, 1))

# Plot counterfactual across area
As = np.linspace(df['A'].min() - 0.05, df['A'].max() + 0.05)

sts.lmplot(
    quap=quapSA, mean_var=quapSA.model.μ,
    x='A', y='L', data=df,
    eval_at={'S': np.ones_like(As), 'A': As},
    ax=axs[1],
)

axs[1].tick_params(left=False)
axs[1].set(title='Counterfactual at S = 1.0 (mean)', ylabel=None)


# ----------------------------------------------------------------------------- 
#         8H4(c) Effect of interaction between M and S
# -----------------------------------------------------------------------------
# TODO move this model to part (b) analysis to show lack of confounding.
# Build 2 models: one without the interaction, and the other with 
with pm.Model():
    M, A, S, L = (pm.MutableData(x, df[x]) for x in list('MASL'))
    α = pm.Normal('α', 1.0, 0.25)
    β_M = pm.Normal('β_M', 0, 0.5)
    β_S = pm.Normal('β_S', 0, 0.5)
    β_A = pm.Normal('β_A', 0, 1)
    μ = pm.Deterministic('μ', α + β_M*M + β_S*S + β_A*A)
    σ = pm.Exponential('σ', 1)
    y = pm.Normal('y', μ, σ, observed=L, shape=S.shape)
    quapMS = sts.quap(data=df)

print('L ~ M + S + A')
sts.precis(quapMS)

# Build a model with an interaction term
with pm.Model():
    M, A, S, L = (pm.MutableData(x, df[x]) for x in list('MASL'))
    α = pm.Normal('α', 1.0, 0.25)
    β_M = pm.Normal('β_M', 0, 0.5)
    β_S = pm.Normal('β_S', 0, 0.5)
    β_A = pm.Normal('β_A', 0, 1)
    β_MS = pm.Normal('β_MS', 0, 0.25)
    μ = pm.Deterministic('μ', α + β_M*M + β_S*S + β_MS*M*S + β_A*A)
    σ = pm.Exponential('σ', 1)
    y = pm.Normal('y', μ, σ, observed=L, shape=S.shape)
    quapMSi = sts.quap(data=df)

print('L ~ M + S + M*S + A')
sts.precis(quapMSi)

# Compare the four models
models = [quapMA, quapSA, quapMS, quapMSi]
mnames = ['M + A', 'S + A', 'M + S + A', 'M + S + M*S + A']

sts.plot_coef_table(sts.coef_table(models, mnames), fignum=6)

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
