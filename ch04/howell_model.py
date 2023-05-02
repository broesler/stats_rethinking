#!/usr/bin/env python3
# =============================================================================
#     File: howell_model.py
#  Created: 2019-07-16 21:56
#   Author: Bernie Roesler
#
r"""
Description: Section 4.3.

Build a model of the distribution of human heights:

    ..math::
        h_i \sim \mathcal{N}(\mu, \sigma)  \text{likelihood}
        \mu \sim \mathcal{N}(178, 20)      \text{mean prior}
        \sigma \sim \mathcal{U}(0, 50)     \text{std prior}

Heights are in [cm].
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats

import stats_rethinking as sts

# Set to True for "Overthinking" (R code 4.23 - 4.25)
SAMPLE_SIZE_FLAG = False  # if True, take only 20 data points

plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(56)  # initialize random number generator

# -----------------------------------------------------------------------------
#        Load Dataset
# -----------------------------------------------------------------------------
data_path = '../data/'

# df: height [cm], weight [kg], age [int], male [0,1]
df = pd.read_csv(data_path + 'Howell1.csv')

if SAMPLE_SIZE_FLAG:
    df = df.sample(n=20)  # small sample size

# Filter adults only
adults = df.loc[df['age'] >= 18]

# Inspect the data
col = 'height'

fig = plt.figure(1, clear=True, constrained_layout=True)
ax = fig.add_subplot()
sts.norm_fit(adults[col], ax=ax)
ax.set(title='Raw data',
       xlabel=col,
       ylabel='density')

# -----------------------------------------------------------------------------
#        Build a model
# -----------------------------------------------------------------------------
# Assume: h_i ~ N(mu, sigma)
# Specify the priors for each parameter:  (R code 4.12, 4.13)
mu_c = 178  # [cm] mean for the height-mean prior
mus_c = 20  # [cm] std  for the height-mean prior
sig_c = 50  # [cm] maximum value for height-stdev prior
mu = stats.norm(mu_c, mus_c)
sigma = stats.uniform(0, sig_c)  # sigma must be positive!

# Sample from the joint prior distribution (R code 4.14)
N = 10_000
sample_mu = mu.rvs(N)
sample_sigma = sigma.rvs(N)
prior_h = stats.norm(sample_mu, sample_sigma)
sample_h = prior_h.rvs(N)

# Compare to a wider distribution (R code 4.15)
sample_mu_wide = stats.norm(178, 100).rvs(N)
sample_h_wide = stats.norm(sample_mu_wide, sample_sigma).rvs(N)

WADLOW_HEIGHT = 272  # tallest man ever

# -----------------------------------------------------------------------------
#        Prior-Predictive Simulation
# -----------------------------------------------------------------------------
# Combine data for plotting convenience
priors = dict(mu=dict(dist=mu, lims=(100, 250),
                      title=rf"$\mu \sim \mathcal{{N}}({mu_c}, {mus_c})$"),
              sigma=dict(dist=sigma, lims=(-10, 60),
                         title=rf"$\sigma \sim \mathcal{{U}}(0, {sig_c})$"),
              )

# Plot the priors (Figure 4.3)
fig = plt.figure(2, clear=True, constrained_layout=True)
gs = fig.add_gridspec(nrows=2, ncols=len(priors))

for i, (name, d) in enumerate(priors.items()):
    x = np.linspace(d['lims'][0], d['lims'][1], 100)
    ax = fig.add_subplot(gs[0, i])
    ax.plot(x, d['dist'].pdf(x))
    ax.set(title=d['title'],
           xlabel=rf"$\{name}$",
           ylabel='density')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

# Plot with the desired joint prior
ax = fig.add_subplot(gs[1, 0])
sts.norm_fit(sample_h, ax=ax)
ax.axvline(sample_h.mean(), c='k', ls='--', lw=1)
ax.set(xlim=(0, 350),
       title=r"$h \sim \mathcal{N}(\mu, \sigma)$",
       xlabel=col,
       ylabel='density')
ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

# Plot with a poor joint prior
ax = fig.add_subplot(gs[1, 1])
sts.norm_fit(sample_h_wide, ax=ax)
ax.axvline(sample_h_wide.mean(), c='k', ls='--', lw=1)
ax.axvline(0.0, c='k', ls='-.', lw=1)
ax.axvline(WADLOW_HEIGHT, c='k', ls='-', lw=1)
ax.set(xlim=(-250, 500),
       title=r"""$h \sim \mathcal{N}(\mu, \sigma)$
$\mu \sim \mathcal{N}(178, 100)$""",
       xlabel=col,
       ylabel='density')
ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))


def print_percentiles(test_height):
    """Check the percentile of the tallest man in the world."""
    top_h = stats.percentileofscore(sample_h, test_height)
    top_h_wide = stats.percentileofscore(sample_h_wide, test_height)
    print(f"---------- Percentile of {test_height:.1f} ----------")
    print(f"         N(mu, sigma): {top_h:.2f}")
    print(f"N(N(178, 100), sigma): {top_h_wide:.2f}")


print_percentiles(0.0)
print_percentiles(WADLOW_HEIGHT)
# Output:
# ---------- Percentile of 0.0 ----------
#          N(mu, sigma): 0.00
# N(N(178, 100), sigma): 4.31
# ---------- Percentile of 272.0 ----------
#          N(mu, sigma): 99.25
# N(N(178, 100), sigma): 81.49

# -----------------------------------------------------------------------------
#         4.3.3 Grid approximation of the posterior distribution (R code 4.16)
# -----------------------------------------------------------------------------
# P(h | data) ∝ P(data | h) * P(h)
#             = P(data | h) * P(h | mu, sigma) * P(mu) * P(sigma)
#             = P(data | h) * N(N(m, s), U(0, v))

Np = 200  # number of parameters values to test

if SAMPLE_SIZE_FLAG:
    mu_start, mu_stop = 140, 170
    sigma_start, sigma_stop = 4, 20
else:
    mu_start, mu_stop = 140, 160
    sigma_start, sigma_stop = 4, 9

mu_list = np.linspace(mu_start, mu_stop, Np)
sigma_list = np.linspace(sigma_start, sigma_stop, Np)
post = sts.expand_grid(mu=mu_list, sigma=sigma_list)


def log_likelihood(data):
    r"""Compute the joint (log) probability of the data given each set of
    parameters:

    ..math::
        f(x) = \log(P(data | x)),

    where :math:`x = (\mu, \sigma)`, for example.
    """
    return lambda x: stats.norm(x['mu'], x['sigma']).logpdf(data).sum()


post['log_likelihood'] = post.apply(log_likelihood(adults[col]), axis=1)

# Equivalent code:
# post['log_likelihood'] = post.apply(lambda x: stats.norm.logpdf(adults[col],
#                                     loc=x['mu'], scale=x['sigma']).sum(),
#                                     axis=1)

# Bayes' rule numerator:
#   P(p | data) ∝ P(data | p)*P(p),
# taking advantage of the fact that log(a*b) == log(a) + log(b)
post['prod'] = (post['log_likelihood']
                + mu.logpdf(post['mu'])
                + sigma.logpdf(post['sigma']))

# Un-logify to get the actual posterior probability values
post['posterior'] = np.exp(post['prod'] - post['prod'].max())

# Contour plot of the results (R code 4.17)
xx, yy, zz = (np.reshape(np.asarray(m), (Np, Np))
              for m in [post['mu'], post['sigma'], post['posterior']])

fig = plt.figure(4, figsize=(14, 4), clear=True, constrained_layout=True)
ax = fig.add_subplot()
cs = ax.contour(xx, yy, zz, cmap='viridis')
ax.clabel(cs, inline=1, fontsize=10)
ax.set_title('Contours of Posterior Probability')
ax.set(xlabel=r'$\mu$',
       ylabel=r'$\sigma$',
       aspect='equal')

# -----------------------------------------------------------------------------
#         Sample from the posterior (R code 4.19 - 22)
# -----------------------------------------------------------------------------
Ns = 10_000
samples = post.sample(n=Ns, replace=True, weights='posterior')

# Plot the samples
fig = plt.figure(5, clear=True, constrained_layout=True)
ax = fig.add_subplot()
ax.scatter(samples['mu'], samples['sigma'], alpha=0.4)
ax.set(title='Posterior Samples',
       xlabel=r'$\mu$',
       ylabel=r'$\sigma$',
       aspect='equal')

# Plot the marginal posterior densities of mu and sigma
fig = plt.figure(6, clear=True, constrained_layout=True)
fig.set_size_inches((12, 5), forward=True)
fig.suptitle('Marginal Posterior Density')
gs = fig.add_gridspec(nrows=1, ncols=2)

ax0 = fig.add_subplot(gs[0])
sts.norm_fit(samples['mu'], ax=ax0, hist_kws=dict(bins=45))
ax0.set(xlabel=f"$\mu$",
        ylabel='density')

ax1 = fig.add_subplot(gs[1], sharey=ax0)
sts.norm_fit(samples['sigma'], ax=ax1, hist_kws=dict(bins=45))
ax1.set(xlabel=f"$\sigma$", ylabel=None)
ax1.tick_params(axis='y', labelleft=False)

# NOTE az.hdi cannot accept a DataFrame/Series, accepts the values only.
print('---------- HPDI of Posterior Samples ----------')
sts.hpdi(samples['mu'], verbose=True)
sts.hpdi(samples['sigma'], verbose=True)

plt.ion()
plt.show()

# =============================================================================
# =============================================================================
