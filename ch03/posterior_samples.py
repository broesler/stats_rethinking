#!/usr/bin/env python3
#==============================================================================
#     File: posterior_samples.py
#  Created: 2019-06-23 23:16
#   Author: Bernie Roesler
#
"""
  Description: Example sampling from a posterior distribution
"""
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pymc3 as pm

from matplotlib.gridspec import GridSpec
from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-darkgrid')
np.random.seed(56)  # initialize random number generator

k = 6       # successes
n = 9       # trials
Np = 1000   # size of parameter grid

# prior: P(p) ~ U(0, 1)
p_grid, posterior, prior = sts.grid_binom_posterior(Np, k, n, prior_func=lambda p: np.ones(p.shape))

# Sample the posterior distribution
Ns = 100_000
samples = np.random.choice(p_grid, p=posterior, size=Ns, replace=True)

## Exact analytical posterior for comparison
Beta = stats.beta(k+1, n-k+1)  # Beta(\alpha = 1, \beta = 1) == U(0, 1)

## Intervals of defined boundaries
precision = 4
width = precision + 2  # only need room for "0.", values \in [0, 1]
fstr = f"{width}.{precision}f"

print(f"----------Beta({k}, {n}) sample----------")
value = np.sum(posterior[p_grid < 0.5])
print(f"P(p < 0.5) = {value:{fstr}}  # Sum the grid search posterior")

value = np.sum(samples < 0.5) / Ns
print(f"P(p < 0.5) = {value:{fstr}}  # Sum the posterior samples")

value = np.sum((samples > 0.5) & (samples < 0.75)) / Ns
print(f"P(0.5 < p < 0.75) = {value:{fstr}}")

## Intervals of defined probability mass
sts.get_quantile(samples, 0.8)
sts.get_quantile(samples, (0.1, 0.9))

#------------------------------------------------------------------------------ 
#        Plot the posterior samples
#------------------------------------------------------------------------------
# Figure 3.1
fig = plt.figure(1, figsize=(8, 4), clear=True)
gs = GridSpec(nrows=1, ncols=2)

# Plot actual sample values
ax1 = fig.add_subplot(gs[0])
ax1.plot(samples, '.', markeredgewidth=0, alpha=0.1)
ax1.set(xlabel='Sample number',
        ylabel='$p$')

# Plot distribution of samples
ax2 = fig.add_subplot(gs[1])
sns.distplot(samples, ax=ax2)
ax2.set(xlabel='$p$',
        ylabel=f"$P(p | k={k}, n={n})$")

gs.tight_layout(fig)

# Figure 3.2
fig = plt.figure(2, clear=True)
gs = GridSpec(nrows=2, ncols=2)

# 1st row: defined boundaries
# 2nd row: defined probability masses
indices = np.array([[p_grid < 0.5,
                      ((p_grid > 0.5) & (p_grid < 0.75))],
                    [p_grid < Beta.ppf(0.80),
                      ((p_grid > Beta.ppf(0.10)) & (p_grid < Beta.ppf(0.90)))]
                   ])

titles = np.array([['$p < 0.50$', '$0.50 < p < 0.75$'],
                   ['lower 80%',  'middle 80%']])

xticks = [0, 0.25, 0.5, 0.75, 1.0]  # custom axis labels

for i in range(2):
    for j in range(2):
        ax = fig.add_subplot(gs[i,j])

        # Plot the exact analytical distribution
        ax.plot(p_grid, Beta.pdf(p_grid),
                        c='k', lw=1, label='Beta$(k+1, n-k+1)$')
        ax.set(xlabel='$p$',
                       ylabel='Density')

        # Fill in the percentiles
        idx = indices[i,j]
        ax.fill_between(p_grid[idx], Beta.pdf(p_grid[idx]), alpha=0.5)
        sts.annotate(titles[i,j], ax)

        ax.set(xticks=xticks, xticklabels=(str(x) for x in xticks))

gs.tight_layout(fig)

#------------------------------------------------------------------------------ 
#        Plot a highly skewed distribution
#------------------------------------------------------------------------------
_, skewed_posterior, _ = sts.grid_binom_posterior(Np, k=3, n=3)
skewed_samples = np.random.choice(p_grid, p=skewed_posterior, size=Ns, replace=True)

print('----------Beta(3, 3) sample----------')
percentile = 0.50  # [percentile] confidence interval
print('Middle 50% PI:')
sts.get_percentiles(skewed_samples, q=percentile)
print('HDPI 50%:')
sts.get_quantile(skewed_samples, q=percentile, q_func=pm.stats.hpd)

# plt.show()
#==============================================================================
#==============================================================================
