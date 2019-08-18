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
ax.set(xlim=(0, 1.05*df['weight'].max()),
       xlabel='weight [kg]',
       ylabel='height [cm]')
ax.legend()

#------------------------------------------------------------------------------ 
#        Build a Polynomial Model of Height
#------------------------------------------------------------------------------
# Standardize the input
w_z = sts.standardize(df['weight'])

# Figure 4.11
Np = 3  # max nummber of polynomial terms
titles = dict({1: 'Linear', 2: 'Quadratic', 3: 'Cubic'})

fig = plt.figure(2, clear=True, figsize=(max(Np*4, 8), 6))
gs = GridSpec(nrows=1, ncols=Np)

for poly_order in range(1, Np+1):
    # poly_order = 1

    with pm.Model() as poly_model:
        # TODO rewrite as a vector with lognorm for beta[0]
        alpha = pm.Normal('alpha', mu=178, sigma=20)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=poly_order)
        # sigma = pm.Uniform('sigma', 0, 50, testval=9)  # unstable with MAP
        sigma = pm.HalfNormal('sigma', sigma=50)  # choose wide Normal instead

        # Polynomial weights:
        #   mu = alpha + w_m[0] * beta[0] + ... + w_m[n] * beta[n]
        w_m = sts.poly_weights(w_z, poly_order)
        mu = pm.Deterministic('mu', alpha + pm.math.dot(beta, w_m))

        # Likelihood
        h = pm.Normal('h', mu=mu, sigma=sigma, observed=df['height'])

        # NOTE MAP fails for poly_order == 1:
        # * "ValueError: Domain error in arguments." at `sts.sample_quap()` because
        #   the quap values for sigma are NaNs.
        # * quap works fine for poly_order > 1
        # * pm.sample() works fine for any poly_order
        #     - all parameters are normally distributed
        # * sigma MAP value seems to be pegged at either end of Uniform or
        #   middle, true value is ~9 for poly_order == 1.
        # * works for testval={8, 9}, but not other values
        quap = sts.quap()
        post = sts.sample_quap(quap, Ns)
        # post = pm.sample(Ns)

    tr = sts.sample_to_dataframe(post).filter(regex='^(?!mu)')
    # tr = pm.trace_to_dataframe(post).filter(regex='^(?!mu)')

    print(f"---------- poly order: {poly_order} ----------")
    print(sts.precis(tr))

    # Sample from normalized inputs
    x = np.arange(0, 71)  # [kg] range of weight inputs
    z = sts.standardize(x, df['weight'])
    z_m = sts.poly_weights(z, poly_order)

    mu_samp = post['alpha'][:, np.newaxis] + np.dot(post['beta'][:, np.newaxis], z_m)
    mu_mean = mu_samp.mean(axis=0)  # [cm] mean height estimate vs. weight

    q = 0.89  # CI interval probability
    h_samp = stats.norm(mu_samp.T, post['sigma']).rvs().T
    h_hpdi = sts.hpdi(h_samp, q=q)

    print('plotting...')
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
    gs.tight_layout(fig)

#------------------------------------------------------------------------------ 
#        DEBUG:
#------------------------------------------------------------------------------
# from pprint import pprint
# with poly_model:
#     map_est = pm.find_MAP()
# print('map_est:')
# pprint(map_est)

# Plot parameter distributions
# fig = plt.figure(3, clear=True)
# gs = GridSpec(nrows=1, ncols=poly_order+2)
# for i, col in enumerate(tr.columns):
#     ax = fig.add_subplot(gs[i])  # left side plot
#     sns.distplot(tr[col], fit=stats.norm)
#     mu, sigma = stats.norm.fit(tr[col])
#     print(f"{col}: mu = {mu:.4f}, sigma = {sigma:.4f}")
# gs.tight_layout(fig)

# TODO
# * test regular pm.Normal() in linear model
# * try HalfNormal for sigma --> how to define halfnorm such that probability
#   for a given value == specified value??

plt.show()

#==============================================================================
#==============================================================================
