#!/usr/bin/env python3
# =============================================================================
#     File: waffledivorce.py
#  Created: 2023-05-04 13:05
#   Author: Bernie Roesler
#
"""
Description: Chapter 5.1 code.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from pathlib import Path
from scipy import stats

import stats_rethinking as sts

plt.ion()
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(56)  # initialize random number generator

# -----------------------------------------------------------------------------
#        Load Dataset (R code 5.1)
# -----------------------------------------------------------------------------
data_path = Path('../data/')
data_file = Path('WaffleDivorce.csv')

df = pd.read_csv(data_path / data_file)

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 50 entries, 0 to 49
# Data columns (total 15 columns):
#    Column             Non-Null Count  Dtype
# ---  ------             --------------  -----
#  0   Location           50 non-null     object
#  1   Loc                50 non-null     object
#  2   Population         50 non-null     float64
#  3   MedianAgeMarriage  50 non-null     float64
#  4   Marriage           50 non-null     float64
#  5   Marriage SE        50 non-null     float64
#  6   Divorce            50 non-null     float64
#  7   Divorce SE         50 non-null     float64
#  8   WaffleHouses       50 non-null     int64
#  9   South              50 non-null     int64
#  10  Slaves1860         50 non-null     int64
#  11  Population1860     50 non-null     int64
#  12  PropSlaves1860     50 non-null     float64
#  13  A                  50 non-null     float64
#  14  D                  50 non-null     float64
# dtypes: float64(9), int64(4), object(2)
# memory usage: 6.0 KB

# Standardize the data
df['A'] = sts.standardize(df['MedianAgeMarriage'])
df['D'] = sts.standardize(df['Divorce'])

print(f"{df['MedianAgeMarriage'].std() = }")

# Define the model (m5.1, R code 5.3)
with pm.Model() as the_model:
    ind = pm.MutableData('ind', df['A'])
    obs = pm.MutableData('obs', df['D'])
    alpha = pm.Normal('alpha', 0, 0.2)
    beta = pm.Normal('beta', 0, 0.5)
    sigma = pm.Exponential('sigma', 1)
    mu = pm.Deterministic('mu', alpha + beta * ind)
    D = pm.Normal('D', mu, sigma, observed=obs, shape=ind.shape)
    quapA = sts.quap()

# Sample the prior predictive lines over 2 standard deviations (R code 5.4)
N_lines = 50
A = np.r_[-2, 2]
with the_model:
    pm.set_data({'ind': A})
    prior = pm.sample_prior_predictive(N_lines)

# Plot prior predictive lines
fig, ax = plt.subplots(num=1, clear=True, constrained_layout=True)
ax.set(xlabel='Median Age Marriage [std]',
       ylabel='Divorce rate [std]',
       aspect='equal')

# Each column of data is a line
ax.plot(np.tile(A, (N_lines, 1)).T, prior.prior['mu'].mean('chain').T,
        'k', alpha=0.4)

# Compute percentile interval of mean (R code 5.5)
A_seq = np.linspace(-3, 3.2, 30)

postA = quapA.sample()
mu_samp = postA['alpha'].values + postA['beta'].values * A_seq[:, np.newaxis]
mu_mean = mu_samp.mean(axis=1)
q = 0.89
mu_pi = sts.percentiles(mu_samp, q=q, axis=1)  # 0.89 default

A_seq = sts.unstandardize(A_seq, df['MedianAgeMarriage'])
mu_mean = sts.unstandardize(mu_mean, df['Divorce'])
mu_pi = sts.unstandardize(mu_pi, df['Divorce'])

# Make the RHS plot (R code 5.5)
fig = plt.figure(2, clear=True, constrained_layout=True)
gs = fig.add_gridspec(nrows=1, ncols=2)
ax = fig.add_subplot(gs[1])  # right-hand plot
ax.scatter('MedianAgeMarriage', 'Divorce', data=df, alpha=0.4)
ax.plot(A_seq, mu_mean, 'C0', label='MAP Prediction')
ax.fill_between(A_seq, mu_pi[0], mu_pi[1],
                facecolor='k', alpha=0.3, interpolate=True,
                label=rf"{100*q:g}% Percentile Interval of $\mu$")
ax.set(xlabel='Median Age Marriage [yr]')
ax.tick_params(axis='y', labelleft=None)

# Repeat the model for marriage rate (m5.2, R code 5.6)
df['M'] = sts.standardize(df['Marriage'])

with the_model:
    pm.set_data({'ind': df['M']})
    quapM = sts.quap()

A_seq = np.linspace(-3, 3.2, 30)

postM = quapM.sample()
mu_samp = postM['alpha'].values + postM['beta'].values * A_seq[:, np.newaxis]
mu_mean = mu_samp.mean(axis=1)
mu_pi = sts.percentiles(mu_samp, q=q, axis=1)  # 0.89 default

A_seq = sts.unstandardize(A_seq, df['Marriage'])
mu_mean = sts.unstandardize(mu_mean, df['Divorce'])
mu_pi = sts.unstandardize(mu_pi, df['Divorce'])

ax = fig.add_subplot(gs[0], sharey=ax)
ax.scatter('Marriage', 'Divorce', data=df, alpha=0.4)
ax.plot(A_seq, mu_mean, 'C0', label='MAP Prediction')
ax.fill_between(A_seq, mu_pi[0], mu_pi[1],
                facecolor='k', alpha=0.3, interpolate=True,
                label=rf"{100*q:g}% Percentile Interval of $\mu$")
ax.set(xlabel='Marriage Rate [per 1000]',
       ylabel='Divorce Rate [per 1000]')

# =============================================================================
# =============================================================================
