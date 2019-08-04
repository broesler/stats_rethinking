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
    alpha = pm.Normal('alpha', mu=178, sd=20)       # parameter priors
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.Uniform('sigma', 0, 50)              # std prior
    mu = pm.Deterministic('mu', alpha + beta*(w - wbar))
    h = pm.Normal('h', mu=mu, sd=sigma, observed=adults['height'])

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
def linear_model(w, observed):
    wbar = w.mean()
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=178, sd=20)             # parameter priors
        beta = pm.Lognormal('beta', mu=0, sd=1)               # new prior!
        sigma = pm.Uniform('sigma', 0, 50)                    # std prior
        mu = pm.Deterministic('mu', alpha + beta*(w - wbar))
        h = pm.Normal('h', mu=mu, sd=sigma, observed=observed)
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
    trace = pm.sample(Ns)

    # NOTE error at sts.quap line:
    # /Users/bernardroesler/miniconda3/envs/dev/lib/python3.7/site-packages/theano/gradient.py:589:
    # UserWarning: grad method was asked to compute the gradient with respect
    # to a variable that is not part of the computational graph of the cost, or
    # is used only by a non-differentiable operator: alpha
    # quap = sts.quap(dict(alpha=alpha, beta=beta, sigma=sigma))
    # tr = sts.sample_quap(quap, Ns)

tr = pm.trace_to_dataframe(trace).filter(regex='^(?!mu)')  # ignore mu
print(sts.precis(tr))
print('cov:')
print(tr.cov())

#------------------------------------------------------------------------------ 
#        Posterior Prediction
#------------------------------------------------------------------------------
# Figure 4.6
with the_model:
    map_est = pm.find_MAP()
ax_d.plot(w, map_est['mu'], 'k', label='MAP Prediction')
ax_d.legend()

# Plot the posterior prediction vs N data points
N_test = [10, 50, 150, adults.shape[0]]
N_lines = 20

fig = plt.figure(3, clear=True)
gs = GridSpec(nrows=2, ncols=2)

for i, N in enumerate(N_test):
    df_n = adults[:N]
    w = df_n['weight']

    the_model = linear_model(w, observed=df_n['height'])
    with the_model:
        # Sample the posterior distributions of parameters
        # quap = sts.quap(dict(alpha=alpha, beta=beta, sigma=sigma))
        # post = sts.sample_quap(quap, 20)  # only sample 20 lines
        trace = pm.sample(Ns)
        post_samp = pm.sample_posterior_predictive(trace, N_lines, 
                                                   vars=[the_model.mu])

    # Plot the raw data
    ax = fig.add_subplot(gs[i])
    ax.get_shared_x_axes().join(ax)
    ax.get_shared_y_axes().join(ax)

    # Plot the raw data
    ax.scatter(df_n['weight'], df_n['height'], alpha=0.5, label='Raw Data')

    # linear model (input pts) x (# curves)
    # TODO rewrite quap/sample_quap to accomodate mu which is shape (Nd,)
    # Pandas tries to impose an index on Series, which fails when we try to do
    # broadcast operations with a numpy array, so need to get the values out
    # model = (post['alpha'].values + post['beta'].values * (w[:, None] - wbar)).T

    for j in range(N_lines):
        # ax.plot(w, model[j, :], 'k-', lw=1, alpha=0.3)
        ax.plot(w, post_samp['mu'][j], 'k-', lw=1, alpha=0.3)
    ax.set(title=f"N = {N}",
           xlabel='weight [kg]',
           ylabel='height [cm]')

gs.tight_layout(fig)

#------------------------------------------------------------------------------ 
#        Plot regression intervals
#------------------------------------------------------------------------------
# tr = sts.sample_quap(quap, Ns)
tr = pm.trace_to_dataframe(trace).filter(regex='^(?!mu)')
mu_at_50 = tr['alpha'] + tr['beta'] * (50 - wbar)

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
# tr = sts.sample_quap(quap, Ns)
x = np.arange(25, 71)
mu_samp = (tr['alpha'].values + tr['beta'].values * (x[:, None] - wbar)).T

# Plot the credible interval for the mean of the height (not including sigma)
q = 0.89
mu_mean = mu_samp.mean(axis=0)  # (Nd,) average mu values for each data point
mu_hpdi = sts.hpdi(mu_samp, q=q)

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
h_samp = stats.norm(mu_samp.T, tr['sigma']).rvs().T
h_hpdi = sts.hpdi(h_samp, q=q)

ax.fill_between(x, h_hpdi[:, 0], h_hpdi[:, 1],
                facecolor='k', alpha=0.2, interpolate=True,
                label=f"{100*q:g}% Credible Interval of Height")
ax.legend()

#==============================================================================
#==============================================================================
