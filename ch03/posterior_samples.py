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

from matplotlib.gridspec import GridSpec
from scipy import stats

from stats_rethinking import utils

plt.style.use('seaborn-darkgrid')
np.random.seed(56)  # initialize random number generator

k = 6       # successes
n = 9       # trials
Np = 1000   # size of parameter grid

prior_func = lambda p: np.ones(p.shape)  # P(p) ~ U(0, 1)
p_grid, posterior, prior = utils.grid_binom_posterior(Np, k, n,
                                                      prior_func=prior_func)

# Sample the posterior distribution
Ns = 100_000
samples = np.random.choice(p_grid, p=posterior, size=Ns, replace=True)

## Intervals of defined boundaries
# Sum the grid search posterior
print(f'P(p < 0.5) = {np.sum(posterior[p_grid < 0.5]):.8f}')
# Sum the posterior samples
print(f'P(p < 0.5) = {np.sum(samples < 0.5) / Ns:.8f}')

## Intervals of defined probability mass
print(f'{80:9d}%')
print(f'{np.quantile(samples, 0.8):10.8f}')
p1, p2 = 0.1, 0.9
print(f'{100*p1:10.0f}% {100*p2:9.0f}%')
with np.printoptions(formatter={'float': '{:10.8f}'.format}):
    print(np.quantile(samples, (0.1, 0.9)))

#------------------------------------------------------------------------------ 
#        Plot the posterior samples
#------------------------------------------------------------------------------
fig = plt.figure(1, clear=True)
gs = GridSpec(nrows=1, ncols=2)
ax1 = fig.add_subplot(gs[1])
sns.distplot(samples, ax=ax1)
ax1.set(xlabel='$p$',
        ylabel='$P(p | \\mathrm{data})$')

ax2 = fig.add_subplot(gs[0])
ax2.plot(samples, '.', markeredgewidth=0, alpha=0.1)
ax2.set(xlabel='Sample number',
        ylabel='$p$')

gs.tight_layout(fig)
plt.show()
#==============================================================================
#==============================================================================
