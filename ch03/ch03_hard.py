#!/usr/bin/env python3
#==============================================================================
#     File: ch03_hard.py
#  Created: 2019-07-01 22:32
#   Author: Bernie Roesler
#
"""
  Description: Solutions to Hard Exercises in Chapter 3.
"""
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.gridspec import GridSpec
from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-darkgrid')
np.random.seed(56)  # initialize random number generator

# Load the given data: 1 == 'boy', 0 == 'girl'
#   birth1[i]: first  child for family i
#   birth2[i]: second child for family i
birth1 = np.array([1,0,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,1,0,
0,0,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,
1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,0,1,1,0,
1,0,1,1,1,0,1,1,1,1])

birth2 = np.array([0,1,0,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,
1,1,1,0,1,1,1,0,1,0,0,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,0,0,1,1,
0,0,0,1,1,1,0,0,0,0])

df = pd.DataFrame(np.vstack([birth1, birth2]).T, columns=['birth1', 'birth2'])

# Compute the posterior distribution for:
#   P(boy | data) ∝ P(data | boy) * P(boy)  
#
Np = 1000                        # [-] size of parameter grid
n = df.size                      # trials
k = np.sum(df.values.flatten())  # total boys

# prior: P(p) ~ U(0, 1); P(data | p) = B(n, p)
p_grid, posterior, prior = sts.grid_binom_posterior(Np, k, n)

# 3H1: MAP estimation
p_max = p_grid[np.argmax(posterior)]
print(f"P(boy | data) = {p_max:10.8f}")

# 3H2: sample from the posterior
Ns = 10_000
samples = np.random.choice(p_grid, p=posterior, size=Ns, replace=True)

hpdi_qs = [0.50, 0.89, 0.97]
hpdi = sts.hpdi(samples, hpdi_qs, width=6, precision=4, verbose=True)

# 3H3
def model_compare(n=0, k=0, p=0.5, ax=None):
    """Plot binomial distribution vs actual data."""
    binom = stats.binom(n=n, p=p).rvs(Ns)  # counts of # boys in n births 
    mode = stats.mode(binom).mode[0]

    # Plot the distribution vs the value from the data
    ax = sns.distplot(binom, label=f"$B({n}, {p:.2f})$")
    ax.axvline(mode, c='C0', ls='--', label=f"Theory: $k = {mode}$")
    ax.axvline(k, c='k', ls='--', label=f"Data: $k = {k}$")
    ax.set(xlabel='Number of Boys', ylabel='Frequency')
    ax.legend()

# Simulate 10,000 replicas of 200 births
p = 0.5  # assume boys are equally likely as girls
fig, ax = plt.subplots(num=1, clear=True)
model_compare(n=n, k=k, p=p, ax=ax)

# 3H4: simulate 10,000 replicas of 100 births (birth1)
fig, ax = plt.subplots(num=2, clear=True)
model_compare(n=birth1.size, k=np.sum(birth1), p=p, ax=ax)

# 3H5: Check assumption that birth1 and birth2 are independent

plt.show()
#==============================================================================
#==============================================================================
