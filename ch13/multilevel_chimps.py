#!/usr/bin/env python3
# =============================================================================
#     File: chimpanzees.py
#  Created: 2023-12-05 10:51
#   Author: Bernie Roesler
#
"""
§13.3 Multilevel chimpanzees with social choice. Logistic Regression.
"""
# =============================================================================

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from pathlib import Path
from scipy import stats
from scipy.special import expit

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
#         Create the model
# -----------------------------------------------------------------------------
# m11.4 + cluster for "block" (R code 13.21)
with pm.Model():
    actor = pm.MutableData('actor', df['actor'])
    treatment = pm.MutableData('treatment', df['treatment'])
    block_id = pm.MutableData('block', df['block'] - 1)
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
    m13_4 = sts.ulam(data=df)

# (R code 13.22)
print('m13.4:')
pt = sts.precis(m13_4)
sts.plot_coef_table(sts.coef_table([m13_4], ['m13.4: block']), fignum=1)

# Plot distributions of deviations by actor and block
fig, ax = plt.subplots(num=2, clear=True)
sns.kdeplot(m13_4.samples['σ_a'].values.flat, bw_adjust=0.5, c='k', label='actor')
sns.kdeplot(m13_4.samples['σ_g'].values.flat, bw_adjust=0.5, c='C0', label='block')
ax.legend()
ax.set(xlabel='Standard Deviation',
       ylabel='Density')


# Model that ignores block, but clusters by actor (R code 13.23)
with pm.Model():
    actor = pm.MutableData('actor', df['actor'])
    treatment = pm.MutableData('treatment', df['treatment'])
    a_bar = pm.Normal('a_bar', 0, 1.5)
    σ_a = pm.Exponential('σ_a', 1)
    a = pm.Normal('a', a_bar, σ_a, shape=(len(df['actor'].unique()),))  # (7,)
    b = pm.Normal('b', 0, 0.5, shape=(len(df['treatment'].unique()),))  # (4,)
    p = pm.Deterministic('p', pm.math.invlogit(a[actor] + b[treatment]))
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m13_5 = sts.ulam(data=df)

# (R code 13.24)
cmp = sts.compare([m13_4, m13_5], mnames=['m13.4: block', 'm13.5: no block'],
                  ic='PSIS', sort=True)

print('ct:')
print(cmp['ct'])

# Model varying effects on the *treatment*  (R code 13.25)
with pm.Model():
    actor = pm.MutableData('actor', df['actor'])
    treatment = pm.MutableData('treatment', df['treatment'])
    block_id = pm.MutableData('block', df['block'] - 1)
    # Hyper-priors
    a_bar = pm.Normal('a_bar', 0, 1.5)
    σ_a = pm.Exponential('σ_a', 1)
    σ_b = pm.Exponential('σ_b', 1)
    σ_g = pm.Exponential('σ_g', 1)
    # Priors
    a = pm.Normal('a', a_bar, σ_a, shape=(len(df['actor'].unique()),))  # (7,)
    b = pm.Normal('b', 0, σ_b, shape=(len(df['treatment'].unique()),))  # (4,)
    g = pm.Normal('g', 0, σ_g, shape=(len(df['block'].unique()),))      # (6,)
    # Linear model
    p = pm.Deterministic(
        'p',
        pm.math.invlogit(a[actor] + g[block_id] + b[treatment])
    )
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m13_6 = sts.ulam(data=df)

# Compare the two models
ct = sts.coef_table([m13_4, m13_6], ['m13.4 (block)', 'm13.6 (block + treatment)'])
print(ct['coef'].unstack('model').filter(like='b[', axis='rows'))

# -----------------------------------------------------------------------------
#         Handling divergences
# -----------------------------------------------------------------------------
# 1. Try resampling m13.4 for funsies.
# m13_4b = sts.ulam(model=m13_4.model, target_accept=0.99)
# idata = pm.sample(model=m13_4.model, target_accept=0.99)
# print('Divergences: ', int(idata.sample_stats['diverging'].sum()))
# print(f"Acceptance: {float(idata.sample_stats['acceptance_rate'].mean()):.2f}")

a ~ Normal(a_bar, σ_a) -> z ~ Normal(0, 1), a = a_bar + σ_a*z

# 2. Non-centered model
with pm.Model():
    actor = pm.MutableData('actor', df['actor'])
    treatment = pm.MutableData('treatment', df['treatment'])
    block_id = pm.MutableData('block', df['block'] - 1)
    # Hyper-priors
    a_bar = pm.Normal('a_bar', 0, 1.5)
    σ_a = pm.Exponential('σ_a', 1)
    σ_g = pm.Exponential('σ_g', 1)
    # Priors
    z = pm.Normal('z', 0, 1, shape=(len(df['actor'].unique()),))  # (7,)
    b = pm.Normal('b', 0, 0.5, shape=(len(df['treatment'].unique()),))  # (4,)
    x = pm.Normal('x', 0, 1, shape=(len(df['block'].unique()),))      # (6,)
    # Linear model
    p = pm.Deterministic(
        'p',
        pm.math.invlogit(a_bar + z[actor]*σ_a + x[block_id]*σ_g + b[treatment])
    )
    pulled_left = pm.Binomial('pulled_left', 1, p, observed=df['pulled_left'])
    m13_4nc = sts.ulam(data=df)

# Summary ess_bulk for m13_4 and m13_4nc
neff = pd.DataFrame(dict(
    neff_c=az.summary(m13_4.samples)['ess_bulk'],
    neff_nc=(
        az.summary(m13_4nc.samples)['ess_bulk']
        .rename(lambda x: x.replace('z', 'a'))
        .rename(lambda x: x.replace('x', 'g'))
    )
))
neff['diff'] = neff['neff_nc'] - neff['neff_c']
print(neff)

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
