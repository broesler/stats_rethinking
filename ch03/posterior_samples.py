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

plt.style.use('seaborn-darkgrid')
np.random.seed(56)  # initialize random number generator

k = 6
n = 9
Np = 1000

p_grid = np.linspace(0, 1, Np)                 # array of parameter values
prob_p = np.ones(Np)                           # P(p) ~ U(0, 1)
prob_data = stats.binom.pmf(k, n, p=p_grid)    # P(data | p) ~ B(n, k)
posterior_u = prob_data * prob_p               # P(p | data)
posterior = posterior_u / np.sum(posterior_u)

# Sample the posterior distribution
Ns = 100_000
samples = np.random.choice(p_grid, p=posterior, size=Ns, replace=True)

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
