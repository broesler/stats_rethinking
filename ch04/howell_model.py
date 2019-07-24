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
from scipy.interpolate import griddata
from matplotlib.gridspec import GridSpec

import stats_rethinking as sts

# plt.ion()
plt.style.use('seaborn-darkgrid')
np.random.seed(56)  # initialize random number generator

#------------------------------------------------------------------------------ 
#        Load Dataset
#------------------------------------------------------------------------------
data_path = '../data/'

# df: height [cm], weight [kg], age [int], male [0,1]
df = pd.read_csv(data_path + 'Howell1.csv')

# Filter adults only
adults = df.loc[df['age'] >= 18]

# Inspect the data
data_col = 'height'

plt.figure(1, clear=True)
ax = sns.distplot(adults[data_col], fit=stats.norm)
ax.set(title='Raw data',
       xlabel=data_col, 
       ylabel='density')

#------------------------------------------------------------------------------ 
#        Build a model
#------------------------------------------------------------------------------
# Assume: h_i ~ N(mu, sigma)
# Specify the priors for each parameter:
mu_c = 178  # [cm] chosen mean for the height-mean prior
mus_c = 20  # [cm] chosen std  for the height-mean prior
sig_c = 50  # [cm] chosen maximum value for height-stdev prior
mu = stats.norm(mu_c, mus_c)
sigma = stats.uniform(0, sig_c)

# Combine data for plotting convenience
priors = dict({'mu':    {'dist': mu,    'lims': (100, 250)},
               'sigma': {'dist': sigma, 'lims': (-10, 60)}})

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

# Plot the sampled joint prior
plt.figure(3, clear=True)
ax = sns.distplot(sample_h, fit=stats.norm)
ax.axvline(sample_h.mean(), c='k', ls='--', lw=1)
ax.set(title='Joint Prior: $h \sim \mathcal{N}(\mu, \sigma)$',
       xlabel=data_col, 
       ylabel='density')

#------------------------------------------------------------------------------ 
#         Grid approximation of the posterior distribution
#------------------------------------------------------------------------------
Np = 200  # number of parameters values to test
# return a DataFrame of the input data
post = sts.expand_grid(mu=np.linspace(140, 160, Np),
                       sigma=np.linspace(4, 9, Np))

# Compute the joint (log) probability of the data given each set of parameters:
#     f(x) = P(data | x),
# where x = (mu, sigma), for example.
def log_likelihood(data):
    return lambda x: stats.norm(x['mu'], x['sigma']).logpdf(data).sum()

post['log_likelihood'] = post.apply(log_likelihood(adults[data_col]), axis=1)

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
cs = plt.contour(xx, yy, zz)
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
ax.axis('equal')
ax.set(title='Posterior Samples',
       xlabel='$\mu$',
       ylabel='$\sigma$')

# Plot the marginal posterior densities of mu and sigma
fig = plt.figure(6, clear=True)
gs = GridSpec(nrows=1, ncols=2)
for i, col in enumerate(['mu', 'sigma']):
    ax = fig.add_subplot(gs[i])
    sns.distplot(samples[col])
    ax.set(xlabel=f"$\\{col}$",
           ylabel='density')
plt.title('Marginal Posterior Density')
gs.tight_layout(fig)

plt.show()
#==============================================================================
#==============================================================================
