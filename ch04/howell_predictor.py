#!/usr/bin/env python3
# =============================================================================
#     File: howell_predictor.py
#  Created: 2019-07-25 21:26
#   Author: Bernie Roesler
#
"""
  Description: Make a linear model of the Howell data. §4.4.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(5656)  # initialize random number generator

# -----------------------------------------------------------------------------
#        Load Dataset
# -----------------------------------------------------------------------------
data_path = '../data/'

# df: height [cm], weight [kg], age [int], male [0,1]
df = pd.read_csv(data_path + 'Howell1.csv')

# Filter adults only
adults = df[df['age'] >= 18]

weight = adults['weight']  # [kg] independent variable
wbar = weight.mean()

Ns = 10_000  # general number of samples to use

# Plot the raw data
fig = plt.figure(1, clear=True, constrained_layout=True)
ax_d = fig.add_subplot()
ax_d.scatter(adults['weight'], adults['height'], alpha=0.5, label='Raw Data')
ax_d.set(title='Adult height vs. weight',
         xlabel='weight [kg]',
         ylabel='height [cm]')

# -----------------------------------------------------------------------------
#        Build a Model
# -----------------------------------------------------------------------------
# Section 4.4.1 model description (R code 4.38)
with pm.Model() as first_model:
    # Define the model
    alpha = pm.Normal('alpha', mu=178, sigma=20)    # parameter priors
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.Uniform('sigma', 0, 50)              # std prior
    mu = pm.Deterministic('mu', alpha + beta*(weight - wbar))
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=adults['height'])

# -----------------------------------------------------------------------------
#        Prior Predictive Simulation (Figure 4.5)
# -----------------------------------------------------------------------------
# Plot the mean height vs weight as predicted by the priors (without the data)
fig = plt.figure(2, clear=True, constrained_layout=True)
gs = fig.add_gridspec(nrows=1, ncols=2)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharex=ax0, sharey=ax0)
for ax in [ax0, ax1]:
    ax.axhline(0, c='k', ls='--', lw=1)   # x-axis
    ax.axhline(272, c='k', ls='-', lw=1)  # Wadlow line
    ax.set(xlim=(weight.min(), weight.max()),
           ylim=(-100, 400),
           xlabel='weight [kg]')

# Plot the model of mean height vs. weight (R code 4.38)
N = 100
with first_model:
    prior_samp = pm.sample_prior_predictive(N)

# (R code 4.39)
ax0.plot(weight, prior_samp.prior['mu'].squeeze('chain').T, 'k', alpha=0.2)

ax0.set(title=r"""A poor prior
$\beta \sim \mathcal{N}(0, 10)$""",
        ylabel='height [cm]')


# Restrict beta to positive values in the new model (R code 4.41 - 4.42)
def define_linear_model(x, y):
    """Define a linear model with the given data."""
    with pm.Model() as model:
        ind = pm.MutableData('ind', x)
        obs = pm.MutableData('obs', y)
        alpha = pm.Normal('alpha', mu=178, sigma=20)  # parameter priors
        beta = pm.LogNormal('beta', mu=0, sigma=1)    # new prior!
        sigma = pm.Uniform('sigma', 0, 50)            # std prior
        # NOTE the mean-shift must be a function of the data on which the model
        # is trained, *not* the data on which it makes predictions!!
        mu = pm.Deterministic('mu', alpha + beta*(ind - np.mean(x)))
        # likelihood -- same shape as the independent variable!
        h = pm.Normal('h', mu=mu, sigma=sigma, observed=obs, shape=ind.shape)
    return model


# Create the linear model ("m4.3" in R code 4.42)
the_model = define_linear_model(x=weight, y=adults['height'])

# Sample the prior and plot (like R Code 4.38 - 4.39, but with new model)
with the_model:
    prior_samp = pm.sample_prior_predictive(N)

ax1.plot(weight, prior_samp.prior['mu'].squeeze('chain').T, 'k', alpha=0.2)

ax1.set(title=r"""A better prior
$\log \beta \sim \mathcal{N}(0, 1)$""")
ax1.tick_params(axis='y', labelleft=False)

# -----------------------------------------------------------------------------
#        Approximate the Posterior Distribution
# -----------------------------------------------------------------------------
# Compute the quadratic approximation to the posterior (R code 4.42)
quap = sts.quap(model=the_model)

# Sample the posterior
post = quap.sample(Ns)
sts.precis(post)
print('covariance:')
print(quap.cov)

# -----------------------------------------------------------------------------
#        Posterior Prediction
# -----------------------------------------------------------------------------
# Figure 4.6 (R code 4.46)
ax_d.plot(weight, quap.map_est['mu'], 'k', label='MAP Prediction')
ax_d.legend()

# Plot the posterior prediction vs N data points (R code 4.48 - 4.49)
N_test = [10, 50, 150, adults.shape[0]]
N_lines = 20


def post_mu(w, alpha, beta):
    """Compute linear model of 'mu'."""
    w = xr.DataArray(w, dims='N' if np.shape(w) else None)
    res = alpha + beta * (w - wbar_n)
    return res


# Figure 4.7
fig = plt.figure(3, clear=True, constrained_layout=True)
gs = fig.add_gridspec(nrows=2, ncols=2)

for i, N in enumerate(N_test):
    df_n = adults[:N]
    w = df_n['weight']
    wbar_n = w.mean()

    # MAP estimate of the parameters
    the_model = define_linear_model(x=w, y=df_n['height'])
    quap = sts.quap(model=the_model)

    # Sample the posterior distributions of parameters
    post = quap.sample(N_lines)  # only sample 20 lines

    # Manually calculate the deterministic variable from the MAP estimates
    post_mu_calc = post_mu(w, post['alpha'], post['beta'])  # (N_lines, N)

    # Plot the raw data
    sharex = ax if i > 0 else None
    sharey = sharex
    ax = fig.add_subplot(gs[i], sharex=sharex, sharey=sharey)

    # Plot the raw data
    ax.scatter(df_n['weight'], df_n['height'], alpha=0.5, label='Raw Data')

    # Plot N_lines approximations
    for j in range(N_lines):
        ax.plot(w, post_mu_calc[j], 'k-', lw=1, alpha=0.3)

    ax.plot(w, quap.map_est['mu'], 'C3-', label='MAP Estimate')

    ax.legend()
    ax.set(title=f"N = {N}",
           xlabel='weight [kg]',
           ylabel='height [cm]')

# -----------------------------------------------------------------------------
#        Plot regression intervals
# -----------------------------------------------------------------------------
# Get larger number of samples for regression interval calcs (R code 4.50)
post = quap.sample(Ns)

# Calculate mu for a single weight input
mu_at_50 = post_mu(50, post['alpha'], post['beta'])

# Figure 4.8 (R code 4.50 -- 4.51)
fig = plt.figure(4, clear=True, constrained_layout=True)
ax = sts.norm_fit(mu_at_50)
ax.set(xlabel=r'$\mu | w = 50$ [kg]',
       ylabel='density')

print('mu @ w = 50 [kg]:')
q = 0.89
sts.hpdi(mu_at_50, q=q, verbose=True)

# Manually write code for: (R code 4.53 - 4.54)
#   mu = sts.link(quap, Ns)
# Generate samples, compute model output for even-interval input
x = np.arange(25., 71.)
mu_samp = post_mu(x, post['alpha'], post['beta'])

# (R code 4.56)
# Plot the credible interval for the mean of the height (not including sigma)
mu_mean = mu_samp.mean(axis=0)    # (Nd,) average values for each data point
mu_hpdi = sts.hpdi(mu_samp, q=q, axis=0)  # (Nd, 2)

# Figure 4.10 (R code 4.55, 4.57)
fig = plt.figure(5, clear=True, constrained_layout=True)
ax = fig.add_subplot()
ax.scatter(adults['weight'], adults['height'], alpha=0.5, label='Raw Data')
ax.plot(x, mu_mean, 'C3', label='MAP Estimate')
ax.fill_between(x, mu_hpdi[0], mu_hpdi[1],
                facecolor='k', alpha=0.3, interpolate=True,
                label=rf"{100*q:g}% Credible Interval of $\mu$")
ax.set(xlabel='weight [kg]',
       ylabel='height [cm]')
ax.legend()

# Calculate the prediction interval, including sigma (R code 4.59, 4.60, 4.62)
# Manually write code for:
#   h_samp = sts.sim(the_model, Ns)
h_samp = stats.norm(mu_samp, post['sigma'].values[:, np.newaxis]).rvs()  # (Ns, Nd)
# h_hpdi = sts.hpdi(h_samp, q=q, axis=0)  # (Nd, 2)
h_pi = sts.percentiles(h_samp, q=q, axis=0)  # (2, Nd)

# (R code 4.61)
ax.fill_between(x, h_pi[0], h_pi[1],
                facecolor='k', alpha=0.2, interpolate=True,
                label=f"{100*q:g}% Credible Interval of Height")
ax.set(xlim=(28, 64), ylim=(128, 182))

ax.legend()

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
