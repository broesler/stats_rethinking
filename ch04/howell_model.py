#!/usr/bin/env python3
#==============================================================================
#     File: howell_model.py
#  Created: 2019-07-16 21:56
#   Author: Bernie Roesler
#
"""
  Description: Section 4.3 code
"""
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from matplotlib.gridspec import GridSpec

import stats_rethinking as sts

# Set to True for "Overthinking" R code 4.24
sample_size_flag = False  # if True, take only 20 data points

plt.ion()
plt.style.use('seaborn-darkgrid')
np.random.seed(56)  # initialize random number generator

#------------------------------------------------------------------------------ 
#        Load Dataset
#------------------------------------------------------------------------------
data_path = '../data/'

# df: height [cm], weight [kg], age [int], male [0,1]
df = pd.read_csv(data_path + 'Howell1.csv')

if sample_size_flag:
    df = df.sample(n=20)  # small sample size

# Filter adults only
adults = df.loc[df['age'] >= 18]

# Inspect the data
col = 'height'

plt.figure(1, clear=True)
ax = sns.distplot(adults[col], fit=stats.norm)
ax.set(title='Raw data',
       xlabel=col, 
       ylabel='density')

#------------------------------------------------------------------------------ 
#        Build a model
#------------------------------------------------------------------------------
# Assume: h_i ~ N(mu, sigma)
# Specify the priors for each parameter:
mu_c = 178  # [cm] mean for the height-mean prior
mus_c = 20  # [cm] std  for the height-mean prior
sig_c = 50  # [cm] maximum value for height-stdev prior
mu = stats.norm(mu_c, mus_c)
sigma = stats.uniform(0, sig_c)  # sigma must be positive!

# Combine data for plotting convenience
priors = dict(mu=dict(dist=mu, lims=(100, 250)),
              sigma=dict(dist=sigma, lims=(-10, 60)))

# Plot the priors
fig = plt.figure(2, clear=True)
gs = GridSpec(nrows=1, ncols=len(priors))

for i, (name, d) in enumerate(priors.items()):
    x = np.linspace(d['lims'][0], d['lims'][1], 100)

    ax = fig.add_subplot(gs[i])
    ax.plot(x, d['dist'].pdf(x))
    ax.set(title='prior distribution',
           xlabel=f"$\{name}$",
           ylabel='density')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

gs.tight_layout(fig)

# Sample from the joint prior distribution
N = 10_000
sample_mu = mu.rvs(N)
sample_sigma = sigma.rvs(N)
prior_h = stats.norm(sample_mu, sample_sigma)
sample_h = prior_h.rvs(N)

# Compare to a wider distribution
sample_mu_wide = stats.norm(178, 100).rvs(N)
sample_h_wide = stats.norm(sample_mu_wide, sample_sigma).rvs(N)

wadlow_height = 272  # tallest man ever

#------------------------------------------------------------------------------ 
#        Prior-Predictive Simulation
#------------------------------------------------------------------------------
fig = plt.figure(3, figsize=(10,6), clear=True)
gs = GridSpec(nrows=1, ncols=2)

# Plot with the desired joint prior
ax = fig.add_subplot(gs[0])
sns.distplot(sample_h, fit=stats.norm, ax=ax)
ax.axvline(sample_h.mean(), c='k', ls='--', lw=1)
ax.set(xlim=(0, 350),
       title='$h \sim \mathcal{N}(\mu, \sigma)$',
       xlabel=col, 
       ylabel='density')
ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

# Plot with a poor joint prior
ax = fig.add_subplot(gs[1])
sns.distplot(sample_h_wide, fit=stats.norm, ax=ax)
ax.axvline(sample_h_wide.mean(), c='k', ls='--', lw=1)
ax.axvline(0.0, c='k', ls='-.', lw=1)
ax.axvline(wadlow_height, c='k', ls='-', lw=1)
ax.set(xlim=(-250, 500),
        title="$h \sim \mathcal{N}(\mu, \sigma)$\n"\
              "$\mu \sim \mathcal{N}(178, 100)$",
       xlabel=col, 
       ylabel='density')
ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

gs.tight_layout(fig)

# Check the percentile of the tallest man in the world
def print_percentiles(test_height):
    top_h = stats.percentileofscore(sample_h, test_height)
    top_h_wide = stats.percentileofscore(sample_h_wide, test_height)
    print(f"---------- Percentile of {test_height:.1f} ----------")
    print(f"         N(mu, sigma): {top_h:.2f}")
    print(f"N(N(178, 100), sigma): {top_h_wide:.2f}")

print_percentiles(0.0)
print_percentiles(wadlow_height)

#------------------------------------------------------------------------------ 
#        Grid approximation of the posterior
#------------------------------------------------------------------------------
# P(h | data) ∝ P(data | h) * P(h) = P(data | h) * N(N(m, s), U(0, v))

#------------------------------------------------------------------------------ 
#         Grid approximation of the posterior distribution
#------------------------------------------------------------------------------
Np = 200  # number of parameters values to test
# return a DataFrame of the input data
if sample_size_flag:
    mu_stop, sigma_stop = 170, 20
else:
    mu_stop, sigma_stop = 160, 9

post = sts.expand_grid(mu=np.linspace(140, mu_stop, Np),
                       sigma=np.linspace(4, sigma_stop, Np))

# Compute the joint (log) probability of the data given each set of parameters:
#     f(x) = P(data | x),
# where x = (mu, sigma), for example.
def log_likelihood(data):
    return lambda x: stats.norm(x['mu'], x['sigma']).logpdf(data).sum()

# post['log_likelihood'] = post.apply(log_likelihood(adults[col]), axis=1)
post['log_likelihood'] = post.apply(lambda x: stats.norm.logpdf(adults[col], loc=x['mu'], scale=x['sigma']).sum(),
                                    axis=1)

# Bayes' rule numerator:
#   P(p | data) ∝ P(data | p)*P(p),
# taking advantage of the fact that log(a*b) == log(a) + log(b)
post['prod'] = (post['log_likelihood']
                + mu.logpdf(post['mu'])
                + sigma.logpdf(post['sigma']))

# Un-logify to get the actual posterior probability values
post['posterior'] = np.exp(post['prod'] - post['prod'].max())

# Contour plot of the results
xx, yy, zz = (np.reshape(np.asarray(m), (Np, Np))
                for m in [post['mu'], post['sigma'], post['posterior']])

fig = plt.figure(4, clear=True)
ax = fig.add_subplot(111)
cs = plt.contour(xx, yy, zz, cmap='viridis')
ax.clabel(cs, inline=1, fontsize=10)
ax.set_title('Contours of Posterior')
ax.set_xlabel('$\mu$')
ax.set_ylabel('$\sigma$')

#------------------------------------------------------------------------------ 
#         Sample from the posterior
#------------------------------------------------------------------------------
Ns = 10_000
samples = post.sample(n=Ns, replace=True, weights='posterior') 

# Plot the samples
fig = plt.figure(5, clear=True)
ax = fig.add_subplot(111)
ax.scatter(samples['mu'], samples['sigma'], alpha=0.05)
ax.set(title='Posterior Samples',
       xlabel='$\mu$',
       ylabel='$\sigma$')

# Plot the marginal posterior densities of mu and sigma
fig = plt.figure(6, clear=True)
gs = GridSpec(nrows=1, ncols=2)
for i, c in enumerate(['mu', 'sigma']):
    ax = fig.add_subplot(gs[i])
    sns.distplot(samples[c], fit=stats.norm)
    ax.set(title='Marginal Posterior Density',
           xlabel=f"$\\{c}$",
           ylabel='density')
gs.tight_layout(fig)

#==============================================================================
#==============================================================================
