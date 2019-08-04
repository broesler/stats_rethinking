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

w = adults['weight']  # [kg] independent variable
wbar = w.mean()

Ns = 10_000  # general number of samples to use

# Plot the raw data
fig = plt.figure(1, clear=True)
ax_d = fig.add_subplot()
ax_d.scatter(adults['weight'], adults['height'], alpha=0.5, label='Raw Data')
ax_d.set(xlabel='weight [kg]',
         ylabel='height [cm]')

#------------------------------------------------------------------------------ 
#        Build a Model
#------------------------------------------------------------------------------
# Section 4.4.1 model description:
with pm.Model() as first_model:
    # Define the model
    alpha = pm.Normal('alpha', mu=178, sigma=20)       # parameter priors
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.Uniform('sigma', 0, 50)              # std prior
    mu = pm.Deterministic('mu', alpha + beta*(w - wbar))
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=adults['height'])

#------------------------------------------------------------------------------ 
#        Prior Predictive Simulation (Figure 4.5)
#------------------------------------------------------------------------------
# Plot the mean height vs weight as predicted by the priors (without the data)
fig = plt.figure(2, clear=True)
gs = GridSpec(nrows=1, ncols=2)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharex=ax0, sharey=ax0)
for ax in [ax0, ax1]:
    ax.axhline(0, c='k', ls='--', lw=1)   # x-axis
    ax.axhline(272, c='k', ls='-', lw=1)  # Wadlow line
    ax.set(xlim=(w.min(), w.max()),
           ylim=(-100, 400),
           xlabel='weight [kg]',
           ylabel='height [cm]')

# Plot the model of mean height vs. weight
N = 100
with first_model:
    prior_samp = pm.sample_prior_predictive(N)

for i in range(N):
    ax0.plot(w, prior_samp['mu'][i], 'k', alpha=0.2)
ax0.set_title('A poor prior')

# Restrict beta to positive values in the new model
# TODO use pm.Data() for x and observed, then use pm.set_data() later.
def linear_model(x, observed):
    """Define a pymc3 model with the given data."""
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=178, sigma=20)  # parameter priors
        beta = pm.Lognormal('beta', mu=0, sigma=1)    # new prior!
        sigma = pm.Uniform('sigma', 0, 50)            # std prior
        mu = pm.Deterministic('mu', alpha + beta*(x - x.mean()))
        h = pm.Normal('h', mu=mu, sigma=sigma, observed=observed)  # likelihood
    return model

the_model = linear_model(w, observed=adults['height'])
with the_model:
    prior_samp = pm.sample_prior_predictive(N)

for i in range(N):
    ax1.plot(w, prior_samp['mu'][i], 'k', alpha=0.2)
ax1.set_title('A better prior')
gs.tight_layout(fig)

#------------------------------------------------------------------------------ 
#        Compute the Posterior Distribution
#------------------------------------------------------------------------------
with the_model:
    # Sample the posterior distributions of parameters
    quap = sts.quap()

post = sts.sample_quap(quap, Ns)
tr = sts.sample_to_dataframe(post).filter(regex='^(?!mu)')
print(sts.precis(tr))
print('cov:')
print(tr.cov())

#------------------------------------------------------------------------------ 
#        Posterior Prediction
#------------------------------------------------------------------------------
# Figure 4.6
ax_d.plot(w, quap['mu'].mean(), 'k', label='MAP Prediction')
ax_d.legend()

# Plot the posterior prediction vs N data points
N_test = [10, 50, 150, adults.shape[0]]
N_lines = 20

# FIgure 4.7
fig = plt.figure(3, clear=True)
gs = GridSpec(nrows=2, ncols=2)

for i, N in enumerate(N_test):
    df_n = adults[:N]
    w = df_n['weight']

    the_model = linear_model(w, observed=df_n['height'])
    with the_model:
        # Compute the MAP estimate of the parameters
        quap = sts.quap(var_names=['alpha', 'beta', 'sigma'])

        # Using MCMC sampling: (much slower, but no need to rewrite model)
        # trace = pm.sample(Ns)
        # post_pc = pm.sample_posterior_predictive(trace, sample=N_lines,
        #                                          var_names=['mu'])

    # Sample the posterior distributions of parameters
    post = sts.sample_quap(quap, N_lines)  # only sample 20 lines
    # Manually calculate the deterministic variable from the MAP estimates
    post['mu'] = (post['alpha'] + post['beta'] * (w[:, None] - wbar)).T

    # Plot the raw data
    ax = fig.add_subplot(gs[i])
    ax.get_shared_x_axes().join(ax)
    ax.get_shared_y_axes().join(ax)

    # Plot the raw data
    ax.scatter(df_n['weight'], df_n['height'], alpha=0.5, label='Raw Data')

    # Plot N_lines approximations
    for j in range(N_lines):
        ax.plot(w, post['mu'][j], 'k-', lw=1, alpha=0.3)

    ax.set(title=f"N = {N}",
           xlabel='weight [kg]',
           ylabel='height [cm]')

gs.tight_layout(fig)

#------------------------------------------------------------------------------ 
#        Plot regression intervals
#------------------------------------------------------------------------------
# Get larger number of samples for regression interval calcs
post = sts.sample_quap(quap, Ns)

# Calculate mu for a single weight input
mu_at_50 = post['alpha'] + post['beta'] * (50 - wbar)

# Figure 4.8 (R code 4.50 -- 4.51)
fig = plt.figure(4, clear=True)
ax = sns.distplot(mu_at_50)
ax.set(xlabel='$\mu | w = 50$ [kg]',
       ylabel='density')

print('mu @ w = 50 [kg]:')
sts.hpdi(mu_at_50, q=0.89, verbose=True)

# Manually write code for: 
#   mu = sts.link(linear_model, Ns)
# Generate samples, compute model output for even-interval input
x = np.arange(25, 71)
mu_samp = (post['alpha'] + post['beta'] * (x[:, None] - wbar)).T

# Plot the credible interval for the mean of the height (not including sigma)
q = 0.89
mu_mean = mu_samp.mean(axis=0)  # (Nd,) average mu values for each data point
mu_hpdi = sts.hpdi(mu_samp, q=q)

# Figure 4.10
fig = plt.figure(5, clear=True)
ax = fig.add_subplot()
ax.scatter(adults['weight'], adults['height'], alpha=0.5, label='Raw Data')
ax.plot(x, mu_mean, 'k', label='MAP Estimate')
ax.fill_between(x, mu_hpdi[:, 0], mu_hpdi[:, 1],
                facecolor='k', alpha=0.3, interpolate=True,
                label=f"{100*q:g}% Credible Interval of $\mu$")
ax.set(xlabel='weight [kg]',
       ylabel='height [cm]')
ax.legend()

# Calculate the prediction interval, including sigma
# Manually write code for: 
#   h_samp = sts.sim(linear_model, Ns)
# NOTE weird transpose combo to broadcast correctly into consistent shape
h_samp = stats.norm(mu_samp.T, post['sigma']).rvs().T
h_hpdi = sts.hpdi(h_samp, q=q)

ax.fill_between(x, h_hpdi[:, 0], h_hpdi[:, 1],
                facecolor='k', alpha=0.2, interpolate=True,
                label=f"{100*q:g}% Credible Interval of Height")
ax.legend()

#==============================================================================
#==============================================================================
