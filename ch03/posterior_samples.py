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

from itertools import groupby
from matplotlib.gridspec import GridSpec
from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-darkgrid')
np.random.seed(56)  # initialize random number generator

k = 6       # successes
n = 9       # trials
Np = 1000   # size of parameter grid
given_data = np.array([1, 0, 1, 1, 1, 0, 1, 0, 1])

# prior: P(p) ~ U(0, 1)
p_grid, posterior, prior = sts.grid_binom_posterior(Np, k, n)

# Sample the posterior distribution
Ns = 100_000
samples = np.random.choice(p_grid, p=posterior, size=Ns, replace=True)

## Exact analytical posterior for comparison
Beta = stats.beta(k+1, n-k+1)  # Beta(\alpha = 1, \beta = 1) == U(0, 1)

## Intervals of defined boundaries
precision = 8
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
n = k = 3  # all wins!
_, skewed_posterior, _ = sts.grid_binom_posterior(Np, k=k, n=n)
skewed_samples = np.random.choice(p_grid, p=skewed_posterior, size=Ns, replace=True)

# Analytical distribution for plotting
Beta_skewed = stats.beta(k+1, n-k+1)  # n = k = 3

print('----------Beta(3, 3) sample----------')
percentile = 0.50  # [percentile] confidence interval
print('Middle 50% PI:')
perc_50 = sts.get_percentiles(skewed_samples, q=percentile)
print('HDPI 50%:')
# hpdi_50 = sts.get_quantile(skewed_samples, q=percentile, q_func=pm.stats.hpd)
hpdi_50 = sts.get_hpdi(skewed_samples, q=percentile)

# Figure 3.3
fig = plt.figure(3, figsize=(8,4), clear=True)
gs = GridSpec(nrows=1, ncols=2)

indices_skewed = np.array([(p_grid > perc_50[0]) & (p_grid < perc_50[1]),
                           (p_grid > hpdi_50[0]) & (p_grid < hpdi_50[1])])
titles_skewed = np.array(['50% Percentile Interval', '50% HPDI'])

for i in range(2):
    ax = fig.add_subplot(gs[i])

    # Plot distribution of samples
    ax.plot(p_grid, Beta_skewed.pdf(p_grid), 
            c='k', lw=1, label='Beta$(k+1, n-k+1)$')
    ax.set(xlabel='$p$',
           ylabel=f"$P(p | k={k}, n={n})$")

    # Fill in the percentiles
    idx = indices_skewed[i]
    ax.fill_between(p_grid[idx], Beta_skewed.pdf(p_grid[idx]), alpha=0.5)
    sts.annotate(titles_skewed[i], ax)

    ax.set(xticks=xticks, xticklabels=(str(x) for x in xticks))

gs.tight_layout(fig)

#------------------------------------------------------------------------------ 
#        2.2 Point Estimates
#------------------------------------------------------------------------------
print('----------Point Estimates----------')
p_map = p_grid[skewed_posterior.argmax()]
print(f"MAP estimate of posterior: {p_map:{fstr}}")

# SLOW!!!
# kde = sts.density(skewed_samples, adjust=0.01).pdf(p_grid)
# p_map_kde = p_grid[kde.argmax()]
# print(f"MAP estimate of   samples: {p_map_kde:{fstr}}")

print(f"Mean:   {np.mean(skewed_samples):{fstr}}")
print(f"Median: {np.median(skewed_samples):{fstr}}")
print(f"Mode:   {stats.mode(skewed_samples).mode[0]:{fstr}}")


def loss_func(posterior, p_grid, kind='abs'):
    """Compute the expected loss function."""
    LOSS_FUNCS = dict({'abs':  lambda d: np.sum(posterior * np.abs(d - p_grid)),
                       'quad': lambda d: np.sum(posterior * np.abs(d - p_grid)**2)
                      })
    try:
        _loss_func = LOSS_FUNCS[kind]
    except KeyError:
        raise KeyError(f'The loss function {kind} is not supported!')
    return np.array([_loss_func(x) for x in p_grid])

abs_loss = loss_func(skewed_posterior, p_grid, kind='abs')
quad_loss = loss_func(skewed_posterior, p_grid, kind='quad')

# Figure 3.4
fig = plt.figure(4, figsize=(8,4), clear=True)
gs = GridSpec(nrows=1, ncols=2)

# Plot distribution of samples
ax0 = fig.add_subplot(gs[0])
ax0.plot(p_grid, Beta_skewed.pdf(p_grid), 
         c='k', label=f"Beta$({k+1}, {n-k+1})$")
ax0.set(xlabel='$p$',
        ylabel=f"$P(p | k={k}, n={n})$")

ax0.axvline(np.mean(skewed_samples),         c='C0', ls='--', label='Mean')
ax0.axvline(np.median(skewed_samples),       c='C1', ls='--', label='Median')
ax0.axvline(stats.mode(skewed_samples).mode, c='C2', ls='--', label='Mode')
ax0.legend(loc='upper left')

# Plot expected loss
ax1 = fig.add_subplot(gs[1])

ax1.plot(p_grid, abs_loss, label='Absolute Function')
ax1.scatter(p_grid[abs_loss.argmin()], abs_loss.min(),
            marker='o', s=50, edgecolors='k', facecolors='none')

ax1.plot(p_grid, quad_loss, label='Quadratic Function')
ax1.scatter(p_grid[quad_loss.argmin()], quad_loss.min(),
            marker='o', s=50, edgecolors='k', facecolors='none')

ax1.set(xlabel='decision, $p$',
        ylabel='expected proportional loss')
ax1.set_ylim((0, ax1.get_ylim()[1]))
ax1.legend()

gs.tight_layout(fig)

#------------------------------------------------------------------------------ 
#       3.3 Sampling to simulate predictions
#------------------------------------------------------------------------------
print('----------Sampling for Simulation----------')
# R code 3.22 - 3.25
n = 2
p = 0.7
binom2 = stats.binom(n=n, p=p)   # frozen distribution

print(f"P(X = {0, 1, 2}), X ~ Bin(n={n}, p={p}):")
print(binom2.pmf(range(n+1)))

print(f"X ~ Bin(n={n}, p={p}):")
print(binom2.rvs(1))  # X ~ {0, 1, 2}

N = 100_000
dummy2 = pd.Series(binom2.rvs(N))
print('Histogram:')
print(dummy2.value_counts(ascending=True))

# Use 9 trials instead of 2
n = 9
p = 0.7
dummy9 = stats.binom.rvs(size=N, n=n, p=p)

# Generate posterior predictive distribution
w = stats.binom.rvs(size=Ns, n=n, p=samples)  # [# successes] == k_s

counts = [np.bincount(dummy9), np.bincount(w) / Ns]
titles = [f"$X \sim \mathrm{{Bin}}(n={n}, p={p})$", 'Posterior Predictive']

# Figure 3.5, 3.6: Posterior predictive probability
fig = plt.figure(5, figsize=(8, 4), clear=True)
gs = GridSpec(nrows=1, ncols=2)
for i in range(2):
    ax = fig.add_subplot(gs[i])
    ax.stem(counts[i], basefmt='none', use_line_collection=True)

    xticks = range(n + 1)
    ax.set(xticks=xticks, xticklabels=(str(x) for x in xticks))
    ax.set(xlabel='$X$', ylabel='Counts', title=titles[i])

gs.tight_layout(fig)

#------------------------------------------------------------------------------ 
#        Figure 3.7: Runs and Switches
#------------------------------------------------------------------------------
# Need to simulate each series of n trials, and count:
#   * longest streak of successes
#   * number of switches from success to failure or vice versa

def longest_streak(data, val=1):
    """Return the longest number of consecutive `val` occurrences in `data`."""
    longest, current = 0, 0
    for x in data:
        if x == val:
            current += 1
        else:
            longest = max(longest, current)
            current = 0
    return max(longest, current)

def count_switches(data):
    """Count the number of times consecutive values differ."""
    return np.count_nonzero(np.diff(data))  # same as sum(abs(diff(...)))

streaks, switches = np.empty(Ns, dtype=np.int64), np.empty(Ns, dtype=np.int64)
for i, k_s in enumerate(w):
    # Create n-vector of [0, 1]
    trials = np.concatenate([np.ones(k_s), np.zeros(n - k_s)])
    np.random.shuffle(trials)  # randomize order (independent events)
    streaks[i] = longest_streak(trials)
    switches[i] = count_switches(trials)

counts = [np.bincount(streaks) / Ns, np.bincount(switches) / Ns]
xlabels = ['longest run length', 'switches']

data_vals = [longest_streak(given_data), count_switches(given_data)]

# Figure 3.7
fig = plt.figure(6, figsize=(8, 4), clear=True)
gs = GridSpec(nrows=1, ncols=2)
for i in range(2):
    ax = fig.add_subplot(gs[i])
    # plot simulated counts
    _, stemlines, _ = ax.stem(counts[i],\
                              linefmt='k', 
                              markerfmt='none', 
                              basefmt='none', 
                              use_line_collection=True)
    plt.setp(stemlines, lw=1)

    # highlight the actual data value
    ax.stem([data_vals[i]], [counts[i][data_vals[i]]],
            linefmt='C0', 
            markerfmt='none',
            basefmt='none',
            use_line_collection=True)

    xticks = range(n + 1)
    ax.set(xticks=xticks, xticklabels=(str(x) for x in xticks))
    ax.set(xlabel=xlabels[i], ylabel='Frequencies')

gs.tight_layout(fig)



plt.show()
#==============================================================================
#==============================================================================
