#!/usr/bin/env python3
# =============================================================================
#     File: multicollinearity.py
#  Created: 2023-05-11 21:54
#   Author: Bernie Roesler
#
"""
Description: §6.1 Multicollinearity Examples.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from pathlib import Path
from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

# ----------------------------------------------------------------------------- 
#         6.1.1 Simulated Legs
# -----------------------------------------------------------------------------
# (R code 6.2)
N = 100  # number of individuals
height = stats.norm(10, 2).rvs(N)
leg_prop = stats.uniform(0.4, 0.1).rvs(N)  # leg length as proportion of height
leg_left = leg_prop * height + stats.norm(0, 0.02).rvs(N)  # proportion + error
leg_right = leg_prop * height + stats.norm(0, 0.02).rvs(N)
df = pd.DataFrame(np.c_[height, leg_left, leg_right],
                  columns=['height', 'leg_left', 'leg_right'])

# Model with both legs (R code 6.3)
with pm.Model() as model:
    α = pm.Normal('α', 10, 100)
    βl = pm.Normal('βl', 2, 10)
    βr = pm.Normal('βr', 2, 10)
    μ = pm.Deterministic('μ', α + βl * df['leg_left'] + βr * df['leg_right'])
    σ = pm.Exponential('σ', 1)
    height = pm.Normal('height', μ, σ, observed=df['height'])
    m6_1 = sts.quap(data=df)

print('m6.1:')
sts.precis(m6_1)
sts.plot_coef_table(sts.coef_table([m6_1]), fignum=1)  # (R code 6.4)

# Plot the posterior (R code 6.5)
post = m6_1.sample()

fig = plt.figure(2, clear=True, constrained_layout=True)
gs = fig.add_gridspec(nrows=1, ncols=2)
ax = fig.add_subplot(gs[0])
ax.scatter('βr', 'βl', data=post, alpha=0.1)
ax.set(xlabel=r'$\beta_l$',
       ylabel=r'$\beta_r$')

# Plot the posterior sum of the parameters (R code 6.6)
sum_blbr = post['βl'] + post['βr']
ax = fig.add_subplot(gs[1])
# sts.norm_fit(sum_blbr, ax=ax)
sns.histplot(sum_blbr, kde=True, stat='density', ax=ax)
ax.set(xlabel='sum of βl and βr')

# Just model one leg! (R code 6.7)
with pm.Model() as model:
    α = pm.Normal('α', 10, 100)
    βl = pm.Normal('βl', 2, 10)
    μ = pm.Deterministic('μ', α + βl * df['leg_left'])
    σ = pm.Exponential('σ', 1)
    height = pm.Normal('height', μ, σ, observed=df['height'])
    m6_2 = sts.quap(data=df)

print('m6.2:')
sts.precis(m6_2)

# -----------------------------------------------------------------------------
#        Load Dataset (R code 5.18)
# -----------------------------------------------------------------------------
# data_path = Path('../data/')
# data_file = Path('foxes.csv')

# df = pd.read_csv(data_path / data_file)

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
