#!/usr/bin/env python3
#==============================================================================
#     File: howell_model.py
#  Created: 2019-07-16 21:56
#   Author: Bernie Roesler
#
"""
  Description: Section 4.3 code
"""
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from matplotlib.gridspec import GridSpec

import stats_rethinking as sts

plt.ion()
plt.style.use('seaborn-darkgrid')
np.random.seed(56)  # initialize random number generator

#------------------------------------------------------------------------------ 
#        Load Dataset
#------------------------------------------------------------------------------
data_path = '../data/'

# df: height [cm], weight [kg], age [int], male [0,1]
df = pd.read_csv(data_path + 'Howell1.csv')

# Filter adults only
adults = df[df['age'] >= 18]

# Inspect the data
col = 'height'

plt.figure(1, clear=True)
ax = sns.distplot(adults[col], fit=stats.norm)
ax.set(xlabel=col, 
       ylabel='density')

#------------------------------------------------------------------------------ 
#        Build a model
#------------------------------------------------------------------------------
# Assume: h_i ~ N(mu, sigma)
# Specify the priors for each parameter:
mu = stats.norm(178, 20)  # mu = 178 cm, sigma = 20 cm
sigma = stats.uniform(0, 50)

# Combine data for plotting convenience
priors = dict({'mu':    {'dist': mu,    'lims': (100, 250)},
               'sigma': {'dist': sigma, 'lims': (-10, 60)}})

# Plot the priors
fig = plt.figure(2, clear=True)
gs = GridSpec(nrows=1, ncols=2)

for i, (name, d) in enumerate(priors.items()):
    x = np.linspace(d['lims'][0], d['lims'][1], 100)

    ax = fig.add_subplot(gs[i])
    ax.plot(x, d['dist'].pdf(x))
    ax.set(title='prior distribution',
           xlabel=f"$\{name}$",
           ylabel='density')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

gs.tight_layout(fig)

# Sample from the joint prior distribution
N = 10_000
sample_mu = mu.rvs(N)
sample_sigma = sigma.rvs(N)
prior_h = stats.norm(sample_mu, sample_sigma)
sample_h = prior_h.rvs(N)

# Compare to a wider distribution
sample_mu_wide = stats.norm(178, 100).rvs(N)
sample_h_wide = stats.norm(sample_mu_wide, sample_sigma).rvs(N)

wadlow_height = 272  # tallest man ever

fig = plt.figure(3, figsize=(10,6), clear=True)
gs = GridSpec(nrows=1, ncols=2)

# Plot the desired joint prior
ax = fig.add_subplot(gs[0])
sns.distplot(sample_h, fit=stats.norm, ax=ax)
ax.axvline(sample_h.mean(), c='k', ls='--', lw=1)
ax.set(xlim=(0, 350),
       title='$h \sim \mathcal{N}(\mu, \sigma)$',
       xlabel=col, 
       ylabel='density')
ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

# Plot a poor joint prior
ax = fig.add_subplot(gs[1])
sns.distplot(sample_h_wide, fit=stats.norm, ax=ax)
ax.axvline(sample_h_wide.mean(), c='k', ls='--', lw=1)
ax.axvline(0.0, c='k', ls='-.', lw=1)
ax.axvline(wadlow_height, c='k', ls='-', lw=1)
ax.set(xlim=(-250, 500),
        title="$h \sim \mathcal{N}(\mu, \sigma)$\n"\
              "$\mu \sim \mathcal{N}(178, 100)$",
       xlabel=col, 
       ylabel='density')
ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

gs.tight_layout(fig)

# Check the percentile of the tallest man in the world
def print_percentiles(test_height):
    top_h = stats.percentileofscore(sample_h, test_height)
    top_h_wide = stats.percentileofscore(sample_h_wide, test_height)
    print(f"---------- Percentile of {test_height:.1f} ----------")
    print(f"         N(mu, sigma): {top_h:.2f}")
    print(f"N(N(178, 100), sigma): {top_h_wide:.2f}")

print_percentiles(0.0)
print_percentiles(wadlow_height)

#------------------------------------------------------------------------------ 
#        Grid approximation of the posterior
#------------------------------------------------------------------------------
# P(h | data) ∝ P(data | h) * P(h) = P(data | h) * N(N(m, s), U(0, v))

#==============================================================================
#==============================================================================
