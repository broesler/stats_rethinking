#!/usr/bin/env python3
# =============================================================================
#     File: ch04_hard.py
#  Created: 2023-05-04 00:20
#   Author: Bernie Roesler
#
"""
Description: Solutions to the "Hard" exercises. 4H3.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats

import stats_rethinking as sts

plt.ion()
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(56)  # initialize random number generator

data_path = '../data/'

# df: height [cm], weight [kg], age [int], male [0,1]
df = pd.read_csv(data_path + 'Howell1.csv')

# ----------------------------------------------------------------------------- 
#         4H3. Log model of people's height
# -----------------------------------------------------------------------------
# Plot the raw data, separate adults and children
is_adult = df['age'] >= 18
adults = df[is_adult]
children = df[~is_adult]

fig = plt.figure(1, clear=True, constrained_layout=True)
ax = fig.add_subplot()
ax.scatter(adults['weight'], adults['height'], alpha=0.4, label='Adults')
ax.scatter(children['weight'], children['height'], c='C2', alpha=0.2,
           label='Children')

ax.set(xlabel='weight [kg]',
       ylabel='height [cm]',
       ylim=(45, 185))
ax.legend()

# (a) Define logarithmic model of height ~ weight
weight = df['weight']
height = df['height']
w_bar = weight.mean()

with pm.Model() as the_model:
    alpha = pm.Normal('alpha', 178, 100)
    beta = pm.Normal('beta', 0, 100)
    sigma = pm.Uniform('sigma', 0, 50)
    mu = alpha + beta * np.log(weight)
    h = pm.Normal('h', mu, sigma, observed=height)

    quap = sts.quap()
    post = quap.sample()

sts.precis(quap)

q = 0.97
w = np.logspace(0, np.log10(1.05*weight.max()), 20)

mu_samp = post['alpha'].values + post['beta'].values * np.log(np.c_[w])
mu_mean = mu_samp.mean(axis=1)
mu_hpdi = sts.hpdi(mu_samp, q=q, axis=1)

h_samp = stats.norm(mu_samp, post['sigma']).rvs()
h_mean = h_samp.mean(axis=1)
h_hpdi = sts.hpdi(h_samp, q=q, axis=1)

# Plot predictions with original data
ax.plot(w, mu_mean, 'k', label='MAP Estimate')
ax.fill_between(w, mu_hpdi[0], mu_hpdi[1],
                facecolor='k', alpha=0.3, interpolate=True,
                label=f"{100*q:g}% CI")
ax.fill_between(w, h_hpdi[0], h_hpdi[1],
                facecolor='k', alpha=0.2, interpolate=True,
                label=f"{100*q:g}% CI")
ax.legend()

# =============================================================================
# =============================================================================
