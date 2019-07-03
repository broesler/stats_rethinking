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

# Load the given data: 1 == 'boy', 2 == 'girl'
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
k = np.sum(df.values.flatten())  # boys

# prior: P(p) ~ U(0, 1); P(data | p) = Bin(n, k, p)
p_grid, posterior, prior = sts.grid_binom_posterior(Np, k, n)

# 3H1: MAP estimation
p_max = p_grid[np.argmax(posterior)]
print(f"P(boy | data) = {p_max:10.8f}")

# 3H2: sample from the posterior
Ns = 10_000
samples = np.random.choice(p_grid, p=posterior, size=Ns, replace=True)

hpdi_qs = [0.50, 0.89, 0.97]
hpdi = list()
for q in hpdi_qs:
    hpdi.append(sts.hpdi(samples, q, width=6, precision=4))

#==============================================================================
#==============================================================================
