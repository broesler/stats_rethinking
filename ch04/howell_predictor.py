#!/usr/bin/env python3
#==============================================================================
#     File: howell_predictor.py
#  Created: 2019-07-25 21:26
#   Author: Bernie Roesler
#
"""
  Description: Make a linear model of the Howell data.
"""
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
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
adults = df.loc[df['age'] >= 18]

Ns = 10_000

w = adults['weight']  # [kg] independent variable
wbar = w.mean()

# Plot the raw data
fig = plt.figure(1, clear=True)
ax = fig.add_subplot()
ax.scatter(adults['weight'], adults['height'], alpha=0.5)
ax.set_xlabel('weight [kg]')
ax.set_ylabel('height [cm]')

#------------------------------------------------------------------------------ 
#        Build a Model
#------------------------------------------------------------------------------
# Build the model:
#   w = actual weight data
#   h ~ N(mu, sigma)
#   mu = alpha + beta*(w - w_bar)
#   alpha = N(178, 20)
#   beta = N(0, 10)
#   sigma = U(0, 50)

#------------------------------------------------------------------------------ 
#        Prior Predictive Simulation (Figure 4.5)
#------------------------------------------------------------------------------
# Plot the height as predicted by just the priors (without the data)
fig = plt.figure(2, clear=True)
gs = GridSpec(nrows=1, ncols=2)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
for ax in [ax0, ax1]:
    ax.axhline(0, c='k', ls='--', lw=1)   # x-axis
    ax.axhline(272, c='k', ls='-', lw=1)  # Wadlow line
    ax.set(xlim=(w.min(), w.max()),
           ylim=(-100, 400),
           xlabel='weight [kg]',
           ylabel='height [cm]')

# Plot the model
N = 100
a = stats.norm(178, 20).rvs(N)
b = stats.norm(0, 10).rvs(N)
h_prior = a + b*(w[:, None] - wbar)  # [Nw, N]
for i in range(N):
    ax0.plot(w, h_prior[:, i], 'k', alpha=0.2)

# Restrict beta to positive values
b_pos = stats.lognorm(s=1, scale=1).rvs(N)
h_prior_better = a + b_pos*(w[:, None] - wbar)  # [Nw, N]
for i in range(N):
    ax1.plot(w, h_prior_better[:, i], 'k', alpha=0.2)

# with pm.Model() as linear_model:
#     alpha = pm.Normal('alpha', mu=178, sd=20)           # parameter priors
#     beta = pm.Normal('beta', mu=0, sd=10)
#     sigma = pm.Uniform('sigma', 0, 50)                  # std prior
#     h = pm.Normal('h', mu=alpha + beta*(w - w.mean()),  # likelihood
#                   sd=sigma,
#                   observed=adults['height'])
#     trace = pm.sample(Ns)
# df = pm.trace_to_dataframe(trace)

#==============================================================================
#==============================================================================
