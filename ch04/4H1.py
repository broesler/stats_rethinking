#!/usr/bin/env python3
# =============================================================================
#     File: ch04_hard.py
#  Created: 2023-05-04 00:20
#   Author: Bernie Roesler
#
"""
Solutions to the "Hard" exercises.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from scipy import stats

import stats_rethinking as sts

plt.ion()
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(56)  # initialize random number generator

data_path = '../data/'

# df: height [cm], weight [kg], age [int], male [0,1]
df = pd.read_csv(data_path + 'Howell1.csv')

# ----------------------------------------------------------------------------- 
#         4H1. Unknown weights
# -----------------------------------------------------------------------------
UNK_W = np.r_[46.95, 43.72, 64.78, 32.59, 54.63]

# Plot the raw data, separate adults and children
is_adult = df['age'] >= 18
adults = df[is_adult]
children = df[~is_adult]

fig = plt.figure(1, clear=True, constrained_layout=True)
ax = fig.add_subplot()
ax.scatter(adults['weight'], adults['height'], alpha=0.4, label='Adults')
ax.scatter(children['weight'], children['height'], c='C2', alpha=0.2,
           label='Children')

# plot min/max new weights to determine model to use
ax.axvline(UNK_W.min(), c='k', ls='--', lw=1)
ax.axvline(UNK_W.max(), c='k', ls='--', lw=1)

ax.set(xlim=(0, 1.05*df['weight'].max()),
       xlabel='weight [kg]',
       ylabel='height [cm]')
ax.legend()

# Unknowns fall within range of adults, so safe to use linear model
weight = adults['weight']
height = adults['height']
w_bar = weight.mean()

with pm.Model() as the_model:
    alpha = pm.Normal('alpha', height.mean(), 3*height.std())
    beta = pm.Lognormal('beta', 0, 1)
    sigma = pm.Uniform('sigma', 0, 50)
    mu = alpha + beta * (weight - w_bar)
    h = pm.Normal('h', mu, sigma, observed=height)

    quap = sts.quap()
    post = quap.sample()

mu_samp = post['alpha'] + post['beta'] * (xr.DataArray(UNK_W) - w_bar)
h_samp = stats.norm(mu_samp, post['sigma'].values[:, np.newaxis]).rvs()
h_mean = h_samp.mean(axis=0)
h_hpdi = sts.hpdi(h_samp, q=0.89, axis=0)  # (2, ...)

with pd.option_context('display.float_format', '{:.2f}'.format):
    print(pd.DataFrame(np.c_[UNK_W, h_mean, h_hpdi.T], 
                       columns=['weight', 'height', '|89%', '89%|']))

# Output:
#    weight   height     |89%     89%|
# 0   46.95   156.45   147.96   164.03
# 1   43.72   153.39   145.23   161.56
# 2   64.78   172.45   164.01   180.67
# 3   32.59   143.34   135.17   151.28
# 4   54.63   163.38   155.21   171.43

# Plot predictions with original data
ax.errorbar(UNK_W, h_mean, yerr=np.abs(h_hpdi - h_mean),
            ls='-', lw=1, marker='.', markersize=10, c='C3',
            capsize=4, ecolor='k', elinewidth=1, alpha=0.8,
            label='Predicted Data')
ax.legend()

# =============================================================================
# =============================================================================
