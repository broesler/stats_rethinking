#!/usr/bin/env python3
# =============================================================================
#     File: 13M4.py
#  Created: 2024-02-19 10:04
#   Author: Bernie Roesler
#
"""
13M4 - Cross-classified model for chimpanzee data with Cauchy priors.
"""
# =============================================================================

import arviz as az
import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
# import xarray as xr

from pathlib import Path
# from scipy import stats
# from scipy.special import expit

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

# -----------------------------------------------------------------------------
#         Create a model with Exponential priors
# -----------------------------------------------------------------------------
# m11.4 + cluster for "block" (R code 13.21)
# This model is "cross-classified", since not all actors are within each block.
with pm.Model():
    actor = pm.MutableData('actor', df['actor'])
    treatment = pm.MutableData('treatment', df['treatment'])
    block_id = pm.MutableData('block_id', df['block'] - 1)
    # Hyper-priors
    a_bar = pm.Normal('a_bar', 0, 1.5)
    σ_a = pm.Exponential('σ_a', 1)
    σ_g = pm.Exponential('σ_g', 1)
    # Priors
    a = pm.Normal('a', a_bar, σ_a, shape=(len(df['actor'].unique()),))  # (7,)
    b = pm.Normal('b', 0, 0.5, shape=(len(df['treatment'].unique()),))  # (4,)
    g = pm.Normal('g', 0, σ_g, shape=(len(df['block'].unique()),))      # (6,)
    # Linear model
    p = pm.Deterministic(
        'p',
        pm.math.invlogit(a[actor] + g[block_id] + b[treatment])
    )
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m_exp = sts.ulam(data=df)

# Same model, but with Cauchy priors
with pm.Model():
    actor = pm.MutableData('actor', df['actor'])
    treatment = pm.MutableData('treatment', df['treatment'])
    block_id = pm.MutableData('block_id', df['block'] - 1)
    # Hyper-priors
    a_bar = pm.Normal('a_bar', 0, 1.5)
    σ_a = pm.HalfCauchy('σ_a', 1)
    σ_g = pm.HalfCauchy('σ_g', 1)
    # Priors
    a = pm.Normal('a', a_bar, σ_a, shape=(len(df['actor'].unique()),))  # (7,)
    b = pm.Normal('b', 0, 0.5, shape=(len(df['treatment'].unique()),))  # (4,)
    g = pm.Normal('g', 0, σ_g, shape=(len(df['block'].unique()),))      # (6,)
    # Linear model
    p = pm.Deterministic(
        'p',
        pm.math.invlogit(a[actor] + g[block_id] + b[treatment])
    )
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m_cau = sts.ulam(data=df)

# (R code 13.22)
models = [m_exp, m_cau]
mnames = ['m_exp', 'm_cau']
sts.plot_coef_table(sts.coef_table(models, mnames), fignum=1)

print(sts.compare(models, mnames)['ct'])

neff = pd.DataFrame(dict(
    neff_exp=az.summary(m_exp.samples)['ess_bulk'],
    neff_cau=az.summary(m_cau.samples)['ess_bulk']
))
neff['diff'] = neff['neff_exp'] - neff['neff_cau']
print(neff)


# -----------------------------------------------------------------------------
#         Plot distributions of deviations by actor and block
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(num=2, clear=True)
sns.kdeplot(m_exp.samples['σ_a'].values.flat,
            bw_adjust=0.5, c='k', label='actor exp')
sns.kdeplot(m_exp.samples['σ_g'].values.flat,
            bw_adjust=0.5, c='C0', label='block exp')
sns.kdeplot(m_cau.samples['σ_a'].values.flat,
            bw_adjust=0.5, c='k', ls='--', label='actor Cauchy')
sns.kdeplot(m_cau.samples['σ_g'].values.flat,
            bw_adjust=0.5, c='C0', ls='--',  label='block Cauchy')
ax.legend()
ax.set(xlabel='Standard Deviation',
       ylabel='Density')

# -----------------------------------------------------------------------------
#         Conclusions
# -----------------------------------------------------------------------------
# Cauchy distribution allows for higher variance, so it's not surprising that
# the standard deviations are higher in the Cauchy model. The effective sample
# size is higher for the Cauchy model in the actor parameters, since it allows
# better sampling of the actors with extreme answers (e.g. all left).


plt.ion()
plt.show()

# =============================================================================
# =============================================================================
