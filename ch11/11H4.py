#!/usr/bin/env python3
# =============================================================================
#     File: 11H4.py
#  Created: 2024-01-11 20:52
#   Author: Bernie Roesler
#
"""
Solution to 11H4. Salamanders data.

.. note:: Problem is mislabeled as 10H4 in the digital version.

The model will be of the number of salamanders as a Poisson model.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pathlib import Path
from scipy import stats

import stats_rethinking as sts

df = pd.read_csv(Path('../data/salamanders.csv'))

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 47 entries, 0 to 46
# Data columns (total 4 columns):
#    Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   SITE       47 non-null     int64
#  1   SALAMAN    47 non-null     int64  # counts of salamanders
#  2   PCTCOVER   47 non-null     int64  # percent ground cover
#  3   FORESTAGE  47 non-null     int64  # age of trees in 49-m² plot
# dtypes: int64(4)
# memory usage: 1.6 KB

# Drop SITE 30 -> no salamanders, no coverage, FORESTAGE == 0
df = df.set_index('SITE').drop(30)
df['S'] = df['SALAMAN']
df['P'] = sts.standardize(np.log(df['PCTCOVER']))

S_mean = df['S'].mean()

# Plot prior predictive
xs = np.linspace(0, 10, 200)
α_strong = stats.lognorm.pdf(xs, s=0.5, scale=S_mean)

fig, axs = plt.subplots(ncols=2, num=1, clear=True)
fig.set_size_inches((10, 5), forward=True)
axs[0].plot(xs, α_strong)
axs[0].set_title((
    r'$\lambda = e^\alpha \text{ with } \alpha \sim '
    rf"\mathcal{{N}}({np.log(S_mean):.2f}, 0.5)$"
))
axs[0].set_xlabel('Mean Number of Salamanders')

# Simulate a prior with a slope
N = 50
xs = np.linspace(-2, 2, 200)
α_prior = stats.norm(np.log(S_mean), 0.5).rvs(N)
β_prior = stats.norm(0, 0.5).rvs(N)
axs[1].plot(xs, np.exp(α_prior + β_prior*np.c_[xs]), c='k', alpha=0.3)
axs[1].set(xlabel='log(% cover) [std]',
           ylabel='Number of Salamanders')

# ----------------------------------------------------------------------------- 
#         (a) Model the salamander counts as a Poisson model
# -----------------------------------------------------------------------------
with pm.Model():
    α = pm.Normal('α', np.log(S_mean), 0.5)
    β = pm.Normal('β', 0, 0.1)
    λ = pm.Deterministic('λ', pm.math.exp(α + β * df['P']))
    y = pm.Poisson('y', λ, observed=df['S'])
    # m_quap = sts.quap()
    m_ulam = sts.ulam(data=df)

print('ulam:')
sts.precis(m_ulam)

# prior = pm.sample_prior_predictive(model=m_ulam.model)

# =============================================================================
# =============================================================================
