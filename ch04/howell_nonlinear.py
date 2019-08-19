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
    # Define the model
    with pm.Model() as poly_model:
        # Parameter priors
        alpha = pm.Normal('alpha', mu=178, sigma=20, shape=(1,))
        b1 = pm.Lognormal('b1', mu=0, sigma=1, shape=(1,))  # linear term > 0
        bn = pm.Normal('bn', mu=0, sigma=10, shape=(poly_order-1,))
        beta = pm.Deterministic('beta', pm.math.concatenate([alpha, b1, bn]))

        # sigma = pm.Uniform('sigma', 0, 50, testval=9)  # unstable with MAP
        sigma = pm.HalfNormal('sigma', sigma=25)  # choose wide Normal instead

        # Polynomial weights:
        #   mu = beta[0] * w_m[0] + beta[1] * w_m[1] + ... + w_m[n] * beta[n]
        # where beta[0] == alpha, w_m[0] = [1, ..., 1]
        w_m = sts.poly_weights(w_z, poly_order)  # [poly_order, Nd]
        mu = pm.Deterministic('mu', pm.math.dot(beta, w_m))

        # Likelihood
        h = pm.Normal('h', mu=mu, sigma=sigma, observed=df['height'])

        # Get the posterior approximation
        quap = sts.quap(vars=[beta, sigma, mu])  # ignore a, b1, bn
        post = sts.sample_quap(quap, Ns)

    tr = sts.sample_to_dataframe(post).filter(regex='^(?!mu)')

    print(f"---------- poly order: {poly_order} ----------")
    print(sts.precis(tr))

    # Sample from normalized inputs
    x = np.arange(0, 71)  # [kg] range of weight inputs
    z = sts.standardize(x, df['weight'])
    z_m = sts.poly_weights(z, poly_order)  # (poly_order, x.size)

    # (Ns, x.size) == Ns, poly_order) * (poly_order, x.size)
    mu_samp = np.dot(post['beta'], z_m)
    mu_mean = mu_samp.mean(axis=0)  # [cm] mean height estimate vs. weight

    q = 0.89  # CI interval probability
    h_samp = stats.norm(mu_samp, post['sigma']).rvs()
    h_hpdi = sts.hpdi(h_samp, q=q)

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
# * rewrite [alpha, beta] as a vector with np.ones for beta[0], lognorm for
#   beta[1], norms for rest of betas
# * test regular pm.Normal() in linear model
# * try HalfNormal for sigma --> how to define halfnorm such that probability
#   for a given value == specified value??

#==============================================================================
#==============================================================================
