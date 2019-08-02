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
adults = df[df['age'] >= 18]

Ns = 10_000
Nd = adults.shape[0]

w = adults['weight']  # [kg] independent variable
wbar = w.mean()

# Plot the raw data
fig = plt.figure(1, clear=True)
ax_d = fig.add_subplot()
ax_d.scatter(adults['weight'], adults['height'], alpha=0.5, label='Raw Data')
ax_d.set(xlabel='weight [kg]',
         ylabel='height [cm]')

#------------------------------------------------------------------------------ 
#        Build a Model
#------------------------------------------------------------------------------
# Build the model:
#   w = actual weight data
#   h ~ N(mu, sigma)
#   where:
#       mu = alpha + beta*(w - w_bar)
#       alpha = N(178, 20)
#       beta = N(0, 10)
#       sigma = U(0, 50)

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
h_prior = a + b*(w[:, None] - wbar)  # (Nw, N)
for i in range(N):
    ax0.plot(w, h_prior[:, i], 'k', alpha=0.2)
ax0.set_title('A poor prior')

# Restrict beta to positive values
b_pos = stats.lognorm(s=1, scale=1).rvs(N)
h_prior_better = a + b_pos*(w[:, None] - wbar)  # (Nw, N)
for i in range(N):
    ax1.plot(w, h_prior_better[:, i], 'k', alpha=0.2)
ax1.set_title('A better prior')
gs.tight_layout(fig)

#------------------------------------------------------------------------------ 
#        Compute the Posterior
#------------------------------------------------------------------------------
with pm.Model() as linear_model:
    # Define the model
    alpha = pm.Normal('alpha', mu=178, sd=20)       # parameter priors
    beta = pm.Lognormal('beta', mu=0, sd=1)
    sigma = pm.Uniform('sigma', 0, 50)              # std prior
    h = pm.Normal('h', mu=alpha + beta*(w - wbar),  # likelihood
                  sd=sigma,
                  observed=adults['height'])
    # Sample the posterior distributions of parameters
    quap = sts.quap(dict(alpha=alpha, beta=beta, sigma=sigma))
    tr = sts.sample_quap(quap, Ns)

print(sts.precis(tr))
print('cov:')
print(tr.cov())

#------------------------------------------------------------------------------ 
#        Posterior Prediction
#------------------------------------------------------------------------------
map_est = tr.mean()
a_map = map_est['alpha']
b_map = map_est['beta']
h_pred = a_map + b_map * (w - wbar)

# Figure 4.6
ax_d.plot(w, h_pred, 'k', label='MAP Prediction')
ax_d.legend()

# Plot the posterior prediction vs N data points
N_test = [10, 50, 150, Nd]

fig = plt.figure(3, clear=True)
gs = GridSpec(nrows=2, ncols=2)

for i, N in enumerate(N_test):
    df_n = adults[:N]
    w = df_n['weight']
    wbar = w.mean()

    with pm.Model() as linear_model:
        # Define the model
        alpha = pm.Normal('alpha', mu=178, sd=20)  # parameter priors
        beta = pm.Lognormal('beta', mu=0, sd=1)
        sigma = pm.Uniform('sigma', 0, 50)         # std prior
        mu = alpha + beta*(w - wbar)
        h = pm.Normal('h', mu=mu, sd=sigma,        # likelihood
                      observed=df_n['height'])
        # Sample the posterior distributions of parameters
        quap = sts.quap(dict(alpha=alpha, beta=beta, sigma=sigma))
        tr = sts.sample_quap(quap, Ns)

    post = tr.sample(20)  # plot 20 lines

    # Plot the raw data
    ax = fig.add_subplot(gs[i])
    ax.get_shared_x_axes().join(ax)
    ax.get_shared_y_axes().join(ax)

    ax.scatter(df_n['weight'], df_n['height'], alpha=0.5, label='Raw Data')

    # linear model (input pts) x (# curves)
    model = post['alpha'].values + post['beta'].values * (w[:, None] - wbar)

    for j in range(post.shape[0]):
        ax.plot(w, model[:, j], 'k-', lw=1, alpha=0.3)
    ax.set(title=f"N = {N}",
           xlabel='weight [kg]',
           ylabel='height [cm]')

gs.tight_layout(fig)

#------------------------------------------------------------------------------ 
#        Plot regression intervals
#------------------------------------------------------------------------------
mu_at_50 = tr['alpha'] + tr['beta'] * (50 - wbar)

# Figure 4.8 (R code 4.50 -- 4.51)
fig = plt.figure(4, clear=True)
ax = sns.distplot(mu_at_50)
ax.set(xlabel='$\mu | w = 50$ [kg]',
       ylabel='density')

sts.hpdi(mu_at_50, 0.89, verbose=True)

# mu = sts.link(model, Ns)
# Generate samples, compute model output for even-interval input
tr = sts.sample_quap(quap, Ns)
x = np.arange(25, 71)
mu_samp = tr['alpha'].values + tr['beta'].values * (x[:, None] - wbar)

# Plot the credible interval for the mean of the height (not including sigma)
q = 0.89
mu_mean = mu_samp.mean(axis=1)  # (Nd, 1) average mu values for each data point
mu_hpdi = np.apply_along_axis(lambda row: sts.hpdi(row, q=q), axis=1, arr=mu_samp)

fig = plt.figure(5, clear=True)
ax = fig.add_subplot()
ax.scatter(adults['weight'], adults['height'], alpha=0.5, label='Raw Data')
ax.plot(x, mu_mean, 'k', label='MAP Estimate')
ax.fill_between(x, mu_hpdi[:, 0], mu_hpdi[:, 1],
                facecolor='k', alpha=0.3, interpolate=True,
                label=f"{100*q:g}% Credible Interval")
ax.set(xlabel='weight [kg]',
       ylabel='height [cm]')
ax.legend()

# Calculate the prediction interval, including sigma


#==============================================================================
#==============================================================================
