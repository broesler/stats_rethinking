#!/usr/bin/env python3
# =============================================================================
#     File: ocean_tools.py
#  Created: 2023-12-13 20:25
#   Author: Bernie Roesler
#
"""
§11.2 Poisson models.

Fit the model:

.. math::
    T \sim \mathrm{Poisson}(\lambda)
    \log \lambda = \alpha_{\mathrm{CID}} + \beta_{\mathrm{CID}} \log P
    \alpha ~ \mathcal{N}(3, 0.5)
    \beta ~ \mathcal{N}(0, 0.2)
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

df = pd.read_csv(Path('../data/Kline.csv'))

# >>> df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 10 entries, 0 to 9
# Data columns (total 5 columns):
#    Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   culture      10 non-null     object
#  1   population   10 non-null     int64
#  2   contact      10 non-null     object
#  3   total_tools  10 non-null     int64
#  4   mean_TU      10 non-null     float64
# dtypes: float64(1), int64(2), object(2)
# memory usage: 532.0 bytes

# ----------------------------------------------------------------------------- 
#         Test Priors
# -----------------------------------------------------------------------------
# Test some priors to get a sense of the log scale
xs = np.linspace(0, 100, 200)

# NOTE r::dlnorm(x, meanlog, sdlog) -> s = sdlog, scale = np.exp(meanlog)
α_weak = stats.lognorm.pdf(xs, 10)
α_strong = stats.lognorm.pdf(xs, 0.5, scale=np.exp(3))

fig, ax = plt.subplots(num=1, clear=True)
ax.plot(xs, α_weak, 'k-', label=r'$\alpha \sim \mathcal{N}(0, 10)$')
ax.plot(xs, α_strong, 'C0-', label=r'$\alpha \sim \mathcal{N}(3, 0.5)$')

ax.legend()
ax.set(
    xlabel='Mean number of tools',
    ylabel='Density',
    xlim=(0, 100),
    ylim=(0, 0.08),
)
ax.spines[['top', 'right']].set_visible(False)

# ----------------------------------------------------------------------------- 
#         Plot prior predictive simulations
# -----------------------------------------------------------------------------
N_lines = 100

xs = np.linspace(-2, 2, 200)[:, np.newaxis]
α = stats.norm(3, 0.5).rvs(N_lines)
β_weak = stats.norm(0, 10).rvs(N_lines)
prior_weak = np.exp(α + β_weak * xs)

β_strong = stats.norm(0, 0.2).rvs(N_lines)
prior_strong = np.exp(α + β_strong * xs)

# Creat unstandardized x data
x = np.linspace(np.log(100), np.log(200_000), 200)[:, np.newaxis]
λ = np.exp(α + β_strong * x)

fig, axs = plt.subplots(num=2, nrows=2, ncols=2, sharey='row', clear=True)

axs[0, 0].plot(xs, prior_weak, 'k-', lw=1, alpha=0.4)
axs[0, 0].set(
    title=r'$\beta \sim \mathcal{N}(0, 10)$',
    xlabel='log population [std]',
    ylabel='total tools',
    xlim=(-2, 2),
    ylim=(0, 100),
)
axs[0, 0].set_xticks(np.arange(-2, 3))

axs[0, 1].plot(xs, prior_strong, 'k-', lw=1, alpha=0.4)
axs[0, 1].set(
    title=r'$\beta \sim \mathcal{N}(0, 0.2)$',
    xlabel='log population [std]',
    xlim=(-2, 2),
)
axs[0, 1].set_xticks(np.arange(-2, 3))

# Plot un-standardized x-axis
axs[1, 0].plot(x, λ, 'k-', lw=1, alpha=0.4)
axs[1, 0].set(
    title=r'$\alpha \sim \mathcal{N}(3, 0.5), \beta \sim \mathcal{N}(0, 0.2)$',
    xlabel='log population',
    ylabel='total tools',
    xlim=(x.min(), x.max()),
    ylim=(0, 500),
)

# Plot on standard population scale
axs[1, 1].plot(np.exp(x), λ, 'k-', lw=1, alpha=0.4)
axs[1, 1].set(
    title=r'$\alpha \sim \mathcal{N}(3, 0.5), \beta \sim \mathcal{N}(0, 0.2)$',
    xlabel='population',
    ylabel='total tools',
    xlim=(0, 200_000),
    ylim=(0, 500),
)

# ----------------------------------------------------------------------------- 
#         Build the Model
# -----------------------------------------------------------------------------

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
