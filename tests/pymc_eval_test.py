#!/usr/bin/env python3
# =============================================================================
#     File: pymc_eval_test.py
#  Created: 2023-05-08 14:18
#   Author: Bernie Roesler
#
"""
Description: Test computation of Deterministic variables with quap inputs.
"""
# =============================================================================

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from scipy import stats

import stats_rethinking as sts

plt.style.use('seaborn-v0_8-darkgrid')

# -----------------------------------------------------------------------------
#        Load Dataset
# -----------------------------------------------------------------------------
data_path = '../data/'

# df: height [cm], weight [kg], age [int], male [0,1]
df = pd.read_csv(data_path + 'Howell1.csv')

# Filter adults only
adults = df[df['age'] >= 18]
wbar = adults['weight'].mean()  # fixed parameter in the model

# Create the linear model ("m4.3" in R code 4.42)
with pm.Model() as the_model:
    ind = pm.MutableData('ind', adults['weight'])
    obs = pm.MutableData('obs', adults['height'])
    alpha = pm.Normal('alpha', mu=178, sigma=20)
    beta = pm.LogNormal('beta', mu=0, sigma=1)
    sigma = pm.Uniform('sigma', 0, 50)
    mu = pm.Deterministic('mu', alpha + beta*(ind - wbar))
    # likelihood -- same shape as the independent variable!
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=obs, shape=ind.shape)
    # Compute the quadratic approximation to the posterior (R code 4.42)
    quap = sts.quap()

# Sample the posterior
post = quap.sample()
sts.precis(post)
print('covariance:')
print(post.drop_vars('chain').to_pandas().cov())

q = 0.89

# Manually write code for: (R code 4.53 - 4.54)
#   mu = sts.link(quap, Ns)
x = np.arange(25., 71.)

mu_samp = post['alpha'].values + post['beta'].values * (np.c_[x] - wbar)
mu_mean = mu_samp.mean(axis=1)    # (Nd,) average values for each data point
mu_hpdi = sts.hpdi(mu_samp, q=q, axis=1)  # (2, Nd)

# Calculate the prediction interval, including sigma (R code 4.59, 4.60, 4.62)
# Manually write code for:
#   h_samp = sts.sim(the_model, Ns)
h_samp = stats.norm(mu_samp, post['sigma']).rvs()
h_pi = sts.percentiles(h_samp, q=q, axis=1)  # (2, Nd)

# NOTE Use pm.set_data({'ind': np.arange(25, 71)}) to update the
# model's independent variable values. Then,
#   h_samp = pm.sample_posterior_predictive(trace)
#   h_mean = h_samp.posterior_predictive['h'].mean(('chain', 'draw'))
# The catch: we need a posterior sample `trace`. Since we want the quap, not
# the actual MCMC samples, we can fake the
# arviz.InferenceData structure using the normal
# approximation samples we already have in the post df.
# Needs coordinates: ('chain' = [0], 'draw' = [0, 1, ..., Ns])
#
# See: <https://github.com/rasmusbergpalm/pymc3-quap/blob/main/quap/quap.py>
# for using `arviz.convert_to_inference_data'

tr = az.convert_to_inference_data(post.expand_dims('chain'))

# Compute the posterior sample of mu using quap values of alpha and beta.
with the_model:
    # Actual MCMC sampling of posterior with fit data
    # tr = pm.sample()  

    # Choose the data on which to evaluate the model
    x_s = x
    # x_s = adults['weight'].sort_values()
    pm.set_data({'ind': x_s})

    # Actual MCMC sampling of posterior with new data
    y_samp = pm.sample_posterior_predictive(tr)
    y_mean = y_samp.posterior_predictive['h'].mean(('chain', 'draw'))

# Use quap samples directly
# Manual loop since alpha and beta are 0-D variables in the model.
mu_s = sts.lmeval(quap, 
                  out=quap.model.mu,
                  dist=post,
                  eval_at={'ind': x_s},
                  params=[quap.model.alpha, quap.model.beta])

mu_s_mean = mu_s.mean('draw')
mu_s_hpdi = sts.hpdi(mu_s, q=q, axis=mu_s.get_axis_num('draw'))

assert np.allclose(mu_samp.T, mu_s)

# Figure 4.10 (R code 4.55, 4.57)
fig = plt.figure(1, clear=True, constrained_layout=True)
ax = fig.add_subplot()

ax.scatter(adults['weight'], adults['height'], alpha=0.5, label='Raw Data')

ax.plot(x, mu_mean, 'C3', label='MAP Estimate')
ax.plot(x_s, mu_s_mean, 'C2', label='pymc model eval')
ax.plot(x_s, y_mean, 'C1', label='MCMC Estimate')

ax.fill_between(x, mu_hpdi[0], mu_hpdi[1],
                facecolor='k', alpha=0.3, interpolate=True,
                label=rf"{100*q:g}% Credible Interval of $\mu$")
ax.fill_between(x_s, mu_s_hpdi[0], mu_s_hpdi[1],
                facecolor='C2', alpha=0.3, interpolate=True,
                label=rf"{100*q:g}% Credible Interval of $\mu$")

ax.fill_between(x, h_pi[0], h_pi[1],
                facecolor='k', alpha=0.2, interpolate=True,
                label=f"{100*q:g}% Credible Interval of Height")

ax.set(xlabel='weight [kg]', xlim=(28, 64),
       ylabel='height [cm]', ylim=(128, 182),)
ax.legend()

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
