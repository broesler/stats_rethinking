#!/usr/bin/env python3
# =============================================================================
#     File: ch03_hard.py
#  Created: 2019-07-01 22:32
#   Author: Bernie Roesler
#
"""
  Description: Solutions to Hard Exercises in Chapter 3.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(56)  # initialize random number generator

# Load the given data: 1 == 'boy', 0 == 'girl'
#   birth1[i]: first  child for family i
#   birth2[i]: second child for family i
df = pd.read_csv('./birth_data.csv')
birth1 = df['birth1']
birth2 = df['birth2']
del df

# Compute the posterior distribution for:
#   P(boy | data) ∝ P(data | boy) * P(boy)
#
Np = 1000                        # [-] size of parameter grid
n = birth1.size + birth2.size    # trials
k = birth1.sum() + birth2.sum()  # total boys

# prior: P(p) ~ U(0, 1); P(data | p) = B(n, p)
p_grid, posterior, prior = sts.grid_binom_posterior(Np, k, n)

# 3H1: MAP estimation
p_max = p_grid[np.argmax(posterior)]
print(f"P(boy | data) = {p_max:10.8f}")  # == 0.55455455

# 3H2: sample from the posterior
Ns = 10_000
samples = np.random.choice(p_grid, p=posterior, size=Ns, replace=True)

hpdi_qs = [0.50, 0.89, 0.97]
for q in hpdi_qs:
    sts.hpdi(samples, q, width=6, precision=4, verbose=True)
# Output:
# |  50%   50%|
# 0.5255 0.5726
# |  11%   11%|
# 0.4985 0.6086
# |   3%    3%|
# 0.4785 0.6276


# 3H3
def model_compare(n=0, k=0, p=0.5, ax=None):
    """Plot binomial distribution vs actual data."""
    if ax is None:
        ax = plt.gca()

    binom = stats.binom(n=n, p=p).rvs(Ns)  # counts of # boys in n births
    mode = stats.mode(binom, keepdims=True).mode[0]

    # Plot the distribution vs the value from the data
    counts = np.bincount(binom)
    idx = np.where(counts > 0)[0]  # only want where condition is True
    counts = counts[idx]
    ax.stem(idx, counts,
            basefmt='none', markerfmt='none',
            label=f"$B({n}, {p:.2f})$ | Theory: $k = {mode}$")
    ax.axvline(k, c='C1', ls='--', label=f"Data: $k = {k}$")
    ax.set(xlabel='Number of Boys', ylabel='Frequency')
    ax.legend()


# Simulate 10,000 replicas of 200 births
p = p_max  # use MAP estimate from the data
fig, ax = plt.subplots(num=1, clear=True)
ax.set_title('All 200 births')
model_compare(n=n, k=k, p=p, ax=ax)

# 3H4: simulate 10,000 replicas of 100 births (birth1)
fig, ax = plt.subplots(num=2, clear=True)
ax.set_title('First birth only')
model_compare(n=birth1.size, k=np.sum(birth1), p=p, ax=ax)

# 3H5: Check assumption that birth1 and birth2 are independent
girl_first = birth2[birth1 == 0]
ng = girl_first.size
kg = girl_first.sum()

fig, ax = plt.subplots(num=3, clear=True)
ax.set_title('Second Births after Girls')
model_compare(n=ng, k=kg, p=p, ax=ax)

# Model far underpredicts boys following girls!

plt.ion()
plt.show()
# =============================================================================
# =============================================================================
