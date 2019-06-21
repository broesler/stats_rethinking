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

from matplotlib.gridspec import GridSpec
from scipy import stats

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
    prior_key : str in {'uniform', 'step', 'exp'}, optional, default 'uniform'
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


def summarize(model, prob=0.89, verbose=False):
    """Get the MAP estimate of the parameter mean and other summary statistics.

    Parameters
    ----------
    model : pymc3 model
        Pymc3 model as defined by `with pm.Model() as model:`.
    prob : float \in [0, 1], default=0.89
        Probability interval value.
    verbose : bool, default=False
        If True, print out summary statistics for model parameter.

    Returns
    -------
    mean_p : dict
        Result of `pm.find_MAP()`. Dictionary of values.
    std_p : float
        Standard deviation of model parameter MAP value, assuming the parameter
        is normally distributed.
    ci : (2,) ndarray
        Lower and upper bounds of `prob` percent confidence interval on
        parameter estimate.
    """
    with model:
        mean_p = pm.find_MAP()  # use MAP estimation for mean
        std_p = ((1 / pm.find_hessian(mean_p, vars=[p]))**0.5)[0,0]

        # Calculate 89% percentile interval
        norm = stats.norm(mean_p, std_p)
        z = stats.norm.ppf([(1 - prob)/2, (1 + prob)/2])
        ci = mean_p['p'] + std_p * z

    if verbose:
        print('MAP Estimate')
        print('------------')
        print('  mean   std  5.5%  94.5%')
        print(f"p {mean_p['p']:4.2f}  {std_p:4.2f}  {ci[0]:4.2f}   {ci[1]:4.2f}")

    return mean_p['p'], std_p, ci

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
data = np.repeat((0, 1), (n-k, k))  # actual toss results
np.random.shuffle(data)
assert n == len(data)
assert k == data.sum()

# Define the model
with pm.Model() as normal_approx:
    p = pm.Uniform('p', 0, 1)  # prior distribution of p
    w = pm.Binomial('w', n=n, p=p, observed=k)  # likelihood

mean_p, std_p, ci = summarize(normal_approx, verbose=True)
norm_a = stats.norm(mean_p, std_p)

## Analytical Posterior
beta = stats.beta(k+1, n-k+1)

#------------------------------------------------------------------------------ 
#        Plot Results
#------------------------------------------------------------------------------
fig = plt.figure(1, clear=True)
ax = fig.add_subplot(111)

for i in reversed(range(NN)):
    Np = Nps[i]

    # Generate the posterior samples on a grid of parameter values
    p_grid, posterior, prior = get_binom_posterior(Np, k, n, prior_key=prior_key)
    p_max = p_grid[np.where(posterior == np.max(posterior))]
    p_max = p_max.mean() if p_max.size > 1 else p_max.item()

    # Plot the result
    ax.axvline(p_max, ls='--', lw=1, c=f'C{NN-1-i}')
    ax.plot(p_grid, posterior / posterior.max(), 
            marker='o', markerfacecolor='none', 
            label=f'Np = {Np}, $p_{{max}}$ = {p_max:.2f}')

    ax.set_xlabel('probability of water, $p$')
    ax.set_ylabel('non-normalized posterior probability of $p$')
    ax.grid(True)

# Plot the prior
p_fine = np.linspace(0, 1, num=100)
ax.plot(p_fine, prior(p_fine), '-', c=0.4*np.array([1, 1, 1]), label='prior')

# Plot the normal approximation
ax.plot(p_fine, norm_a.pdf(p_fine) / norm_a.pdf(p_fine).max(),
        'C3', label=f'Quad Approx: $\mu = {mean_p:.2f}, \sigma = {std_p:.2f}$')
ax.plot(p_fine, beta.pdf(p_fine) / beta.pdf(p_fine).max(),
        'k-', label=f'True Posterior: $\\beta({k+1}, {n-k+1})$')

title = '$P \sim $ {}  |  trials: {}, events: {}'\
          .format(PRIOR_D[prior_key]['title'], k, n)
ax.set_title(title)
ax.legend()
plt.tight_layout()
plt.show()

#==============================================================================
#==============================================================================
