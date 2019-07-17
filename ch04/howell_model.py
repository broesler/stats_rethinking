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

data_path = '../data/'
df = pd.read_csv(data_path + 'Howell1.csv')

# Filter adults only
adults = df[df['age'] >= 18]

# Inspect the data
col = 'height'

plt.figure(1, clear=True)
ax = sns.distplot(adults[col], fit=stats.norm)
ax.set(xlabel=col, 
       ylabel='density')

# Build a model
# Assume: h_i ~ N(mu, sigma)
# Specify the priors for each parameter:
mu = stats.norm(178, 20)  # mu = 178 cm, sigma = 20 cm
sigma = stats.uniform(0, 50)

# Plot the priors
x1 = np.linspace(mu.ppf(0.01), mu.ppf(0.99))
x2 = np.linspace(sigma.ppf(0.01), sigma.ppf(0.99))

fig = plt.figure(2, clear=True)
gs = GridSpec(nrows=1, ncols=2)

ax1 = fig.add_subplot(gs[0])
ax1.plot(x1, mu.pdf(x1))
ax1.set(xlabel='$\mu$',
        ylabel='density')
ax1.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

ax2 = fig.add_subplot(gs[1])
ax2.plot(x2, sigma.pdf(x2))
ax2.set(xlabel='$\sigma$')
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

gs.tight_layout(fig)

#==============================================================================
#==============================================================================
