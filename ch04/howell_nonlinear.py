#!/usr/bin/env python3
#==============================================================================
#     File: howell_nonlinear.py
#  Created: 2019-08-01 22:23
#   Author: Bernie Roesler
#
"""
  Description: Build non-linear models of the Howell data
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

Ns = 10_000  # general number of samples to use

# Plot the raw data, separate adults and children
is_adult = df['age'] >= 18
adults = df[is_adult]
children = df[~is_adult]

fig = plt.figure(1, clear=True)
ax = fig.add_subplot()
ax.scatter(adults['weight'], adults['height'], alpha=0.5, label='Adults')
ax.scatter(children['weight'], children['height'], c='C3', alpha=0.5, 
           label='Children')
ax.set(xlabel='weight [kg]',
         ylabel='height [cm]')
ax.legend()

#------------------------------------------------------------------------------ 
#        Build a Polynomial Model of Height
#------------------------------------------------------------------------------
# Standardize the input
w_z = (df['weight'] - df['weight'].mean()) / df['weight'].std() 
# w_z =                    (w - w.mean()) / w.std() == stats.zscore(w, ddof=1)
# w_z = (N / (N-1))**0.5 * (w - w.mean()) / w.std() == stats.zscore(w, ddof=0)

# Figure 4.11
Np = 3  # max nummber of polynomial terms

titles = dict({1: 'Linear', 2: 'Quadratic', 3: 'Cubic'})

fig = plt.figure(2, clear=True)
gs = GridSpec(nrows=1, ncols=Np)

for poly_order in range(1, Np+1):

    with pm.Model() as poly_model:
        sigma = pm.Uniform('sigma', 0, 50)

        alpha = pm.Normal('alpha', 178, 20)
        beta_0 = pm.Lognormal('beta_0', 0, 1)
        if poly_order == 1:
            mu = alpha + beta_0 * w_z
        elif poly_order == 2:
            beta_1 = pm.Normal('beta_1', 0, 1)
            mu = alpha + beta_0 * w_z + beta_1 * w_z**2
        elif poly_order == 3:
            beta_1 = pm.Normal('beta_1', 0, 1)
            beta_2 = pm.Normal('beta_2', 0, 1)
            mu = alpha + beta_0 * w_z + beta_1 * w_z**2 + beta_2 * w_z**3

        h = pm.Normal('h', mu=mu, sd=sigma, observed=df['height'])

        var = dict(sigma=sigma, alpha=alpha, beta_0=beta_0)
        if poly_order > 1:
            var['beta_1'] = beta_1
        if poly_order > 2:
            var['beta_2'] = beta_2

        quap = sts.quap(var)
        tr = sts.sample_quap(quap, Ns)

    print(f"poly order: {poly_order}")
    print(sts.precis(tr))

    # Sample from normalized inputs
    x = np.arange(0, 71)
    z = (x - df['weight'].mean()) / df['weight'].std()

    mu_samp = tr['alpha'].values  # (Ns,)
    for i in range(poly_order):
        # Weird "+=" issue here
        mu_samp = mu_samp + tr[f"beta_{i}"].values * z[:, None]**(i+1)
    mu_mean = mu_samp.mean(axis=1)  # [cm] mean height estimate vs. weight

    q = 0.89
    h_samp = stats.norm(mu_samp, tr['sigma']).rvs()
    h_hpdi = sts.hpdi(h_samp.T, q=q)

    # Plot vs the data (in non-normalized x-axis for readability)
    ax = fig.add_subplot(gs[poly_order-1])
    ax.scatter(df['weight'], df['height'], alpha=0.5, label='Data')
    ax.plot(x, mu_mean, 'k', label='Model')
    ax.fill_between(x, h_hpdi[:, 0], h_hpdi[:, 1],
                    facecolor='k', alpha=0.2, interpolate=True,
                    label=f"{100*q:g}% CI")
    ax.set(title=titles[poly_order],
        xlabel='weight [kg]',
        ylabel='height [cm]')
    ax.legend()

#==============================================================================
#==============================================================================
