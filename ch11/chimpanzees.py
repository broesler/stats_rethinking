#!/usr/bin/env python3
# =============================================================================
#     File: chimpanzees.py
#  Created: 2023-12-05 10:51
#   Author: Bernie Roesler
#
"""
§11.1 Chimpanzees with social choice. Logistic Regression.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pathlib import Path
from scipy import stats
from scipy.special import logit, expit

import stats_rethinking as sts

# Get the data (R code 11.1), forcing certain dtypes
df = pd.read_csv(
    Path('../data/chimpanzees.csv'),
    dtype=dict({
        'actor': int,
        'condition': bool,
        'prosoc_left': bool,
        'chose_prosoc': bool,
        'pulled_left': bool,
    })
)

df['actor'] -= 1  # python is 0-indexed

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 504 entries, 0 to 503
# Data columns (total 8 columns):
#    Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   actor         504 non-null    category
#  1   recipient     252 non-null    float64
#  2   condition     504 non-null    bool
#  3   block         504 non-null    int64
#  4   trial         504 non-null    int64
#  5   prosoc_left   504 non-null    bool
#  6   chose_prosoc  504 non-null    bool
#  7   pulled_left   504 non-null    bool
# dtypes: bool(4), category(1), float64(1), int64(2)
# memory usage: 14.5 KB

# Define a treatment index variable combining others (R code 11.2)
df['treatment'] = df['prosoc_left'] + 2 * df['condition']  # in range(4)

# (R code 11.3)
# print(df.pivot_table(
#     index='treatment',
#     values=['prosoc_left', 'condition'],
#     aggfunc='sum',
# ))

# Build a simple model (R code 11.4)
with pm.Model():
    a = pm.Normal('a', 0, 10)
    p = pm.Deterministic('p', pm.math.invlogit(a))
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m11_1 = sts.quap(data=df)

bad_prior = m11_1.sample_prior(N=10_000).sortby('p')
bad_dens = stats.gaussian_kde(bad_prior['p'], bw_method=0.01).pdf(bad_prior['p'])

# Model with better prior
with pm.Model():
    a = pm.Normal('a', 0, 1.5)
    p = pm.Deterministic('p', pm.math.invlogit(a))
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m11_1 = sts.quap(data=df)

reg_prior = m11_1.sample_prior(N=10_000).sortby('p')
reg_dens = stats.gaussian_kde(reg_prior['p'], bw_method=0.01).pdf(reg_prior['p'])

# ----------------------------------------------------------------------------- 
#         FIgure 11.3 (R code 11.6)
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(num=1, ncols=2, clear=True)
fig.set_size_inches((10, 5), forward=True)

ax = axs[0]

ax.plot(bad_prior['p'], bad_dens, c='k', label=r"$a \sim \mathcal{N}(0, 10)$")
ax.plot(reg_prior['p'], reg_dens, c='C0', label=r"$a \sim \mathcal{N}(0, 1.5)$")

ax.legend()
ax.set(xlabel="prior probability 'pulled_left'",
       ylabel='Density')
ax.spines[['top', 'right']].set_visible(False)


# ----------------------------------------------------------------------------- 
#         Include b effect (R code 11.7)
# -----------------------------------------------------------------------------
with pm.Model():
    a = pm.Normal('a', 0, 1.5)
    b = pm.Normal('b', 0, 10, shape=(4,))
    p = pm.Deterministic('p', pm.math.invlogit(a + b[df['treatment']]))
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m11_2 = sts.quap(data=df)

# TODO not sure why sampling the prior gives a `p_dim_0` = 508. 
# Write into pymc help as to why we need to explicitly compute the function
# that has already been defined in the model.

# Get the difference in treatments (R code 11.8)
bad_prior = m11_2.sample_prior(N=10_000)
p = expit(bad_prior['a'] + bad_prior['b'])
bad_diff = np.abs(p[:, 0] - p[:, 1]).sortby(lambda x: x)
bad_dens = stats.gaussian_kde(bad_diff, bw_method=0.01).pdf(bad_diff)

# More regularizing prior on b (R code 11.9)
with pm.Model():
    a = pm.Normal('a', 0, 1.5)
    b = pm.Normal('b', 0, 0.5, shape=(4,))
    p = pm.Deterministic('p', pm.math.invlogit(a + b[df['treatment']]))
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m11_3 = sts.quap(data=df)

reg_prior = m11_3.sample_prior(N=10_000)
p = expit(reg_prior['a'] + reg_prior['b'])
reg_diff = np.abs(p[:, 0] - p[:, 1]).sortby(lambda x: x)
reg_dens = stats.gaussian_kde(reg_diff, bw_method=0.01).pdf(reg_diff)

# Plot on the right side
ax = axs[1]
ax.plot(bad_diff, bad_dens, c='k', label=r"$b \sim \mathcal{N}(0, 10)$")
ax.plot(reg_diff, reg_dens, c='C0', label=r"$b \sim \mathcal{N}(0, 1.5)$")

ax.legend()
ax.set(xlabel="prior probability of\ndifference between treatments",
       ylabel='Density')
ax.spines[['top', 'right']].set_visible(False)

# ----------------------------------------------------------------------------- 
#         Create the model with actor now (R Code 11.10)
# -----------------------------------------------------------------------------
with pm.Model():
    a = pm.Normal('a', 0, 1.5, shape=(len(df['actor'].unique()),))  # (7,)
    b = pm.Normal('b', 0, 0.5, shape=(len(df['treatment'].unique()),))    # (4,)
    p = pm.Deterministic(
        'p',
        pm.math.invlogit(a[df['actor']] + b[df['treatment']])
    )
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m11_4 = sts.ulam(data=df)

sts.precis(m11_4)

# (R code 11.11)
post = m11_4.get_samples()
p_left = expit(post['a'])
p_left.name = 'p'

fig, ax = sts.plot_precis(p_left, mname='m11_4', fignum=2)

fig.set_size_inches((6, 3), forward=True)
ax.axvline(0.5, ls='--', c='k')

# Plot the treatment effects (R code 11.12)
labels = ['R/N', 'L/N', 'R/P', 'L/P']
fig, ax = sts.plot_precis(post['b'], fignum=3, labels=labels)
fig.set_size_inches((6, 3), forward=True)

# Plot the differences in the treatments
post_b = (
    post['b']
    .assign_coords(b_dim_0=labels)
    .stack(sample=('chain', 'draw'))
    .transpose('sample', ...)
)

diffs = dict(
    dbR=post_b.sel(b_dim_0='R/N') - post_b.sel(b_dim_0='R/P'),
    dbL=post_b.sel(b_dim_0='L/N') - post_b.sel(b_dim_0='L/P')
)
sts.plot_precis(pd.DataFrame(diffs), fignum=4)

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
