#!/usr/bin/env python3
#==============================================================================
#     File: gaussians.py
#  Created: 2019-07-08 23:08
#   Author: Bernie Roesler
#
"""
  Description: Chapter 4 Code
"""
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from matplotlib.gridspec import GridSpec

import stats_rethinking as sts

plt.style.use('seaborn-darkgrid')
np.random.seed(56)  # initialize random number generator

# Demonstrate central limit theorem
N = 1000  # number of players
Ns = 16   # number of coin flips (steps to take)

player = stats.uniform(loc=-1, scale=2)  # U(-1, 1)
players = player.rvs(size=(N, Ns))       # sample for each player, Ns times
# add initial column of 0, and sum
players = np.hstack([np.zeros((N,1)), players]).cumsum(axis=1)

#------------------------------------------------------------------------------ 
#        Figure 4.2
#------------------------------------------------------------------------------
fig = plt.figure(1, clear=True)
gs = GridSpec(2, 3)

ax1 = fig.add_subplot(gs[0, 0:])  # upper row

# only plot 100 lines for proof of concept
for p in players[:100]:
    ax1.plot(p, 'k-', lw=1, alpha=0.1)

# pick random to highlight
idx = np.random.choice(N)
ax1.plot(players[idx], 'C0-', lw=2)

ax1.set(xlabel='step number',
        ylabel='position [m]')

# Plot distribution of final locations for each of N \in {4, 8, 16}
axes = list()
for i, n in enumerate([4, 8, 16]):
    ax = fig.add_subplot(gs[1, i])
    axes.append(ax)
    kde = sts.density(players[:, n])
    ax.plot(kde)

plt.show()
#==============================================================================
#==============================================================================
