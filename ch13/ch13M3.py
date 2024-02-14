#!/usr/bin/env python3
# =============================================================================
#     File: ch13M1-3.py
#  Created: 2024-02-14 16:30
#   Author: Bernie Roesler
#
"""
Ch. 13 Exercise M3.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pymc as pm

from pathlib import Path

import stats_rethinking as sts

df = pd.read_csv(
    Path('../data/reedfrogs.csv'),
    dtype=dict(pred='category', size='category')
)

# Approximate the tadpole mortality in each tank (R code 13.2)
df['tank'] = df.index

# Repeat Gaussian multilevel model (R code 13.3)
with pm.Model():
    N = pm.ConstantData('N', df['density'])
    tank = pm.ConstantData('tank', df['tank'])
    α_bar = pm.Normal('α_bar', 0, 1.5)
    σ = pm.Exponential('σ', 1)
    α = pm.Normal('α', α_bar, σ, shape=tank.shape)
    p = pm.Deterministic('p', pm.math.invlogit(α[tank]))
    S = pm.Binomial('S', N, p, observed=df['surv'])
    m_gauss = sts.ulam(data=df)

# Create Cauchy model
with pm.Model():
    N = pm.ConstantData('N', df['density'])
    tank = pm.ConstantData('tank', df['tank'])
    α_bar = pm.Normal('α_bar', 0, 1.5)
    σ = pm.HalfCauchy('σ', 1)
    α = pm.Cauchy('α', α_bar, σ, shape=tank.shape)
    p = pm.Deterministic('p', pm.math.invlogit(α[tank]))
    S = pm.Binomial('S', N, p, observed=df['surv'])
    m_cauchy = sts.ulam(data=df)


# Compare posterior means of α
sample_dims = ('chain', 'draw')
diff_a = (m_gauss.samples['α'] - m_cauchy.samples['α']).mean(sample_dims)

fig, ax = plt.subplots(num=1, clear=True)

# Label tank sizes
ax.axvline(15.5, ls='--', c='gray', lw=1)
ax.axvline(31.5, ls='--', c='gray', lw=1)
ax.text(     8, 0, 'small tanks (10)',  ha='center', va='bottom', transform=ax.get_xaxis_transform())
ax.text(15 + 8, 0, 'medium tanks (25)', ha='center', va='bottom', transform=ax.get_xaxis_transform())
ax.text(31 + 8, 0, 'large tanks (35)',  ha='center', va='bottom', transform=ax.get_xaxis_transform())

# Plot the errors
ax.scatter(df['tank'], diff_a)

ax.set(xlabel='tank',
       ylabel='α')
ax.spines[['top', 'right']].set_visible(False)


# -----------------------------------------------------------------------------
#         Plot the model predictions
# -----------------------------------------------------------------------------
def plot_model(model, dodge=0, color='C0', label=None, ax=None):
    if ax is None:
        ax = plt.gca()

    p_est = model.deterministics['p'].mean(('chain', 'draw'))
    a = (1 - 0.89) / 2
    p_PI = model.deterministics['p'].quantile([a, 1-a], dim=('chain', 'draw'))

    ax.errorbar(
        df['tank'] + dodge,
        p_est,
        yerr=np.abs(p_PI - p_est),
        marker='o', ls='none', c=color, lw=3, alpha=0.7,
        label=label,
    )

    return ax


fig, ax = plt.subplots(num=2, clear=True)

# Label tank sizes
ax.axvline(15.5, ls='--', c='gray', lw=1)
ax.axvline(31.5, ls='--', c='gray', lw=1)
ax.text(     8, 0, 'small tanks (10)',  ha='center', va='bottom')
ax.text(15 + 8, 0, 'medium tanks (25)', ha='center', va='bottom')
ax.text(31 + 8, 0, 'large tanks (35)',  ha='center', va='bottom')

# Plot the data
ax.scatter('tank', 'propsurv', data=df, c='k', label='data')

# Plot each model
plot_model(m_gauss, dodge=-0.2, color='C0', label='Gauss', ax=ax)
plot_model(m_cauchy, dodge=0.2, color='C3', label='Cauchy', ax=ax)

ax.legend()
ax.set(xlabel='tank',
        ylabel='proportion survival',
        ylim=(-0.05, 1.05))
ax.spines[['top', 'right']].set_visible(False)


# -----------------------------------------------------------------------------
#         Explanation
# -----------------------------------------------------------------------------
# NOTE the large differences occur when propsurv = 1.0!
# The intercept values for the Cauchy distribution are much larger than those
# for the Gaussian distribution in these cases.

# Inspect the observations where the errors differ by a lot:
# [70]>>> df.loc[np.abs(diff_a).values > 1]
# [70]===
#     density pred   size  surv  propsurv  tank
# 1        10   no    big    10       1.0     1
# 3        10   no    big    10       1.0     3
# 6        10   no  small    10       1.0     6
# 19       25   no    big    25       1.0    19
# 37       35   no  small    35       1.0    37

# [71]>>> df.loc[np.abs(diff_a).values < 1]
# [71]===
#     density  pred   size  surv  propsurv  tank
# 0        10    no    big     9  0.900000     0
# 2        10    no    big     7  0.700000     2
# 4        10    no  small     9  0.900000     4
# 5        10    no  small     9  0.900000     5
# 7        10    no  small     9  0.900000     7
# 8        10  pred    big     4  0.400000     8
# 9        10  pred    big     9  0.900000     9
# 10       10  pred    big     7  0.700000    10
# 11       10  pred    big     6  0.600000    11
# 12       10  pred  small     7  0.700000    12
# 13       10  pred  small     5  0.500000    13
# 14       10  pred  small     9  0.900000    14
# 15       10  pred  small     9  0.900000    15
# 16       25    no    big    24  0.960000    16
# 17       25    no    big    23  0.920000    17
# 18       25    no    big    22  0.880000    18
# 20       25    no  small    23  0.920000    20
# 21       25    no  small    23  0.920000    21
# 22       25    no  small    23  0.920000    22
# 23       25    no  small    21  0.840000    23
# 24       25  pred    big     6  0.240000    24
# 25       25  pred    big    13  0.520000    25
# 26       25  pred    big     4  0.160000    26
# 27       25  pred    big     9  0.360000    27
# 28       25  pred  small    13  0.520000    28
# 29       25  pred  small    20  0.800000    29
# 30       25  pred  small     8  0.320000    30
# 31       25  pred  small    10  0.400000    31
# 32       35    no    big    34  0.971429    32
# 33       35    no    big    33  0.942857    33
# 34       35    no    big    33  0.942857    34
# 35       35    no    big    31  0.885714    35
# 36       35    no  small    31  0.885714    36
# 38       35    no  small    33  0.942857    38
# 39       35    no  small    32  0.914286    39
# 40       35  pred    big     4  0.114286    40
# 41       35  pred    big    12  0.342857    41
# 42       35  pred    big    13  0.371429    42
# 43       35  pred    big    14  0.400000    43
# 44       35  pred  small    22  0.628571    44
# 45       35  pred  small    12  0.342857    45
# 46       35  pred  small    31  0.885714    46
# 47       35  pred  small    17  0.485714    47

# =============================================================================
# =============================================================================
