#!/usr/bin/env python3
#==============================================================================
#     File: grid_approx.py
#  Created: 2019-06-17 11:17
#   Author: Bernie Roesler
#
"""
  Description: Grid approximation example.
"""
#==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns

from matplotlib.gridspec import GridSpec
from scipy import stats

plt.style.use('seaborn-darkgrid')
np.random.seed(123)  # initialize random number generator

# Possible prior distributions
PRIOR_D = dict({'uniform': {'prior': lambda p: np.ones(p.shape),
                            'title': '$U(0, 1)$'},
                'step': {'prior': lambda p: np.where(p < 0.5, 0, 1),
                         'title': '0 where $p < 0.5$, 1 otherwise'},
                'exp': {'prior': lambda p: np.exp(-5 * np.abs(p - 0.5)),
                        'title': '$-5e^{{|p - 0.5|}}$'}
                })


def get_binom_posterior(Np, k, n, prior_key='uniform'):
    """Posterior probability assuming a binomial distribution likelihood and
    arbitrary prior.

    Parameters
    ----------
    Np : int
        Number of parameter values to use.
    k : int
        Number of event occurrences observed.
    n : int
        Number of trials performed.
    prior_key : {'uniform' | 'step' | 'exp'}, optional, default 'uniform'
        String describing the desired prior distribution.

    Returns
    -------
    p_grid : (Np, 1) ndarray
        Vector of parameter values.
    posterior : (Np, 1) ndarray
        Vector of posterior probability values.
    """
    p_grid = np.linspace(0, 1, Np)  # vector of possible parameter values
    prior_func = PRIOR_D[prior_key]['prior']
    prior = prior_func(p_grid)
    likelihood = stats.binom.pmf(k, n, p_grid)  # binomial distribution
    posterior = likelihood * prior
    # posterior = posterior_u / np.sum(posterior_u)  # normalize to sum to 1
    return p_grid, posterior, prior_func


#------------------------------------------------------------------------------ 
#        Define Parameters
#------------------------------------------------------------------------------
# Data
k = 6  # number of event occurrences, i.e. "heads"
n = 9  # number of trials, i.e. "tosses"

## Grid-search parameters
prior_key = 'uniform'  # 'uniform', 'step', 'exp'
Nps = [5, 20, 100]  # range of grid sizes to try
NN = len(Nps)

## Compute quadratic approximation
# MAP estimation of the parameter mean
with pm.Model() as normal_approx:
    p = pm.Uniform('p', 0, 1)  # prior distribution of p
    w = pm.Binomial('w', n=n, p=p, observed=k)  # likelihood
    map_est = pm.find_MAP()  # use MAP estimation for mean
    mean_p = map_est['p']  # extract desired value

    # The Hessian of a Gaussian == "precision" == 1 / sigma**2
    std_p = ((1 / pm.find_hessian(map_est, vars=[p]))**0.5)[0,0]

    # Calculate percentile interval, assuming normal distribution
    prob = 0.89
    norm = stats.norm(mean_p, std_p)
    z = stats.norm.ppf([(1 - prob)/2, (1 + prob)/2])
    ci = mean_p + std_p * z

    print('MAP Estimate')
    print('------------')
    print('  mean   std  5.5%  94.5%')
    print(f"p {mean_p:4.2f}  {std_p:4.2f}  {ci[0]:4.2f}   {ci[1]:4.2f}")

# Normal approximation to the posterior
norm_a = stats.norm(mean_p, std_p)

## MCMC estimation of parameter mean
Ns = 1000  # number of samples
p_trace = np.empty(Ns)  # initialize array of samples
p_trace[0] = 0.5
for i in range(1, Ns):
    p_new = stats.norm.rvs(loc=p_trace[i-1], scale=0.1)
    if p_new < 0:
        p_new = np.abs(p_new)
    if p_new > 1:
        p_new = 2 - p_new
    q0 = stats.binom.pmf(k, n, p_trace[i-1])
    q1 = stats.binom.pmf(k, n, p_new)
    t = stats.uniform.rvs()
    p_trace[i] = p_new if t < q1/q0 else p_trace[i-1]

## Analytical Posterior
beta = stats.beta(k+1, n-k+1)

#------------------------------------------------------------------------------ 
#        Plot Results
#------------------------------------------------------------------------------
fig = plt.figure(1, figsize=(8, 6), clear=True)
ax = fig.add_subplot(111)

# Plot grid approximation posteriors
for i, Np in enumerate(reversed(Nps)):
    # Generate the posterior samples on a grid of parameter values
    p_grid, posterior, prior = get_binom_posterior(Np, k, n, prior_key=prior_key)
    p_max = p_grid[np.where(posterior == np.max(posterior))]
    p_max = p_max.mean() if p_max.size > 1 else p_max.item()

    # Plot the result
    ax.axvline(p_max, c=f'C{NN-1-i}', ls='--', lw=1)
    ax.plot(p_grid, posterior / posterior.max(), 
            marker='o', markerfacecolor='none', 
            label=f'Np = {Np}, $p_{{max}}$ = {p_max:.2f}')

p_fine = np.linspace(0, 1, num=100)

# Plot the normal approximation
norm_ap = norm_a.pdf(p_fine) 
ax.plot(p_fine, norm_ap / norm_ap.max(),
        'C3', label=f'Quad Approx: $\mathcal{{N}}({mean_p:.2f}, {std_p:.2f})$')
ax.axvline(p_fine[norm_ap.argmax()], c='C3', ls='--', lw=1)

# Plot the analytical posterior
beta_p = beta.pdf(p_fine)
ax.plot(p_fine, beta_p / beta_p.max(),
        'k-', label=f'True Posterior: $\\beta({k+1}, {n-k+1})$')
ax.axvline(p_fine[beta_p.argmax()], c='k', ls='--', lw=1)

# Plot the MCMC approximation
kde = stats.gaussian_kde(p_trace, bw_method=0.75)
kde_p = kde(p_fine)
ax.plot(p_fine, kde_p / kde_p.max(),
        c='C4', label = 'MCMC Posterior')
ax.axvline(p_fine[kde_p.argmax()], c='C4', ls='--', lw=1)
# sns.kdeplot(p_trace, ax=ax, c='C4', label='MCMC Posterior')

# Plot the prior
# ax.plot(p_fine, prior(p_fine), '-', c=0.4*np.array([1, 1, 1]), label='prior')

# Plot formatting
title = '$P \sim $ {}  |  trials: {}, events: {}'\
          .format(PRIOR_D[prior_key]['title'], k, n)
ax.set_title(title)
ax.set_xlabel('probability of water, $p$')
ax.set_ylabel('non-normalized posterior probability of $p$')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

#==============================================================================
#==============================================================================
