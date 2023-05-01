#!/usr/bin/env python3
# =============================================================================
#     File: gaussians.py
#  Created: 2019-07-08 23:08
#   Author: Bernie Roesler
#
"""
  Description: Chapter 4 Code Sections 4.1 - 4.2
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import seaborn as sns

from scipy import stats

# import stats_rethinking as sts


def norm_fit(data, ax=None):
    """Plot a histogram and a normal curve fit to the data."""
    if ax is None:
        ax = plt.gca()
    sns.histplot(data, stat='density', alpha=0.4, ax=ax)
    norm = stats.norm(data.mean(), data.std())
    x = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 1000)
    y = norm.pdf(x)
    ax.plot(x, y, 'C0')


plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(56)  # initialize random number generator

# Demonstrate central limit theorem (R code 4.1)
N = 1000  # number of players
Ns = 16   # number of coin flips (steps to take)

player = stats.uniform(loc=-1, scale=2)  # U(-1, 1)
players = player.rvs(size=(N, Ns))       # sample for each player, Ns times
# add initial column of 0, and sum
players = np.hstack([np.zeros((N, 1)), players]).cumsum(axis=1)

# -----------------------------------------------------------------------------
#        Figure 4.2
# -----------------------------------------------------------------------------
fig = plt.figure(1, clear=True, constrained_layout=True)
gs = fig.add_gridspec(2, 3)

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
    # Plot line in first subplot
    ax1.axvline(n, c='k', ls='--', lw=1, alpha=0.5)

    # Add secondary subplot
    ax = fig.add_subplot(gs[1, i])
    if i > 0:
        ax.sharex(axes[i-1])
    axes.append(ax)

    # Generate the kernel density estimate, and normal fit
    data = players[:, n]  # select the relevant column
    norm_fit(data, ax=ax)
    ax.set_title(f"{n} steps", fontsize=12)
    ax.set(xlabel='position',
           ylabel='Density')


# -----------------------------------------------------------------------------
#        Gaussian from product of RVs (R code 4.2-4.4)
# -----------------------------------------------------------------------------
N, Np = 10000, 12


# Product of Np samples, repeated N times
def prod_dist(p):
    unif = stats.uniform(0, p)
    return np.prod(1 + unif.rvs(size=(N, Np)), axis=1)


fig = plt.figure(2, figsize=(8, 3), clear=True, constrained_layout=True)
gs = fig.add_gridspec(nrows=1, ncols=3)

for i, p in enumerate([0.01, 0.1, 0.5]):
    ax = fig.add_subplot(gs[i])
    norm_fit(prod_dist(p), ax=ax)
    ax.set(title=f"$p = {p}$",
           xlabel='value',
           ylabel='density')

# log of large deviate
p = 0.5
fig = plt.figure(3, clear=True, constrained_layout=True)
ax = fig.add_subplot()
norm_fit(np.log(prod_dist(p)), ax=ax)
ax.set(title=f"$p = {p}$",
       xlabel='value',
       ylabel='log(density)')

plt.ion()
plt.show()
# =============================================================================
# =============================================================================
